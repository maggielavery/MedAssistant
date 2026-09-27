"""Stage-level evaluation harness for every agent in the workflow.

Each case in ``synthetic_cases.json`` targets one stage and is scored by
deterministic checks (no LLM judge), so a failure names the exact expectation
that was missed. Stages:

    intent      agents.classify_intent
    symptom     agents.extract_symptoms (onset/severity, context, reporter)
    retrieval   retrieval.build_timeline (threshold, patient scope, ordering)
    flag        agents.evaluate_flags (decision + embedding-only confidence)
    action      mocked ActionEvent creation through a full workflow turn
    responder   agents.compose_reply (action wording, safety, formatting)
    summarizer  summarizer.summarize_session (title, sources, idempotency)
    workflow    multi-turn end-to-end behavior (follow-ups, checkup, dedupe)

Every stage calls the live model or embeddings, so an OPENAI_API_KEY is
required. Database-backed stages run in a throwaway test database (migrated and
seeded fresh each run); the development database is never touched.

    python eval/run_eval.py                       # all cases
    python eval/run_eval.py --stage flag,symptom  # selected stages
    python eval/run_eval.py --case weight_gain    # cases whose name contains this
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import django

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "continuum.settings")
django.setup()

from django.core.management import call_command  # noqa: E402
from django.db import connection, transaction  # noqa: E402
from django.utils import timezone  # noqa: E402

from assistant_app import agents, llm, retrieval, summarizer, workflow  # noqa: E402
from assistant_app.models import (  # noqa: E402
    ActionEvent,
    ChatSession,
    Flag,
    Message,
    Patient,
    PatientStatus,
)

CASES_PATH = Path(__file__).resolve().parent / "synthetic_cases.json"
STAGES = (
    "intent", "symptom", "retrieval", "flag",
    "action", "responder", "summarizer", "workflow",
)
ISO_MINUTE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
MARKDOWN = re.compile(r"\*\*|__|^\s*#{1,6}\s|^\s*\|.*\|\s*$|^\s*[-*]\s", re.MULTILINE)
# Claims that a mocked action was completed. "Sent to the care team" is the
# accurate description; booking or delivery to the doctor is not. Negated forms
# such as "hasn't been booked" are excluded by the lookbehinds.
COMPLETED_ACTION = re.compile(
    r"(?<!n't )(?<!not )\b(been|is|was|are) (now )?"
    r"(booked|scheduled|rescheduled|confirmed|delivered)\b"
    r"|\bI(?:'ve| have) (booked|scheduled|rescheduled|confirmed|delivered)\b"
    r"|\b(sent|forwarded|delivered|passed)\b[^.?!]{0,25}\bto (your |the )?(doctor|dr\b|cardiologist)"
    r"|\b(doctor|dr\.? \w+) (has|have) (received|read|seen)\b",
    re.IGNORECASE,
)
# Internal vocabulary that must never reach the patient.
INTERNAL_TERMS = re.compile(r"watch[- ]?for|\b[a-z]+_[a-z_]+\b|\bworkflow\b", re.IGNORECASE)
# Greeting or self-introduction after the session greeting already did that.
REGREET = re.compile(r"^\W*(hi|hello|hey)\b|\bI'?m Maggie\b|\bI am Maggie\b", re.IGNORECASE)
# Asking the patient to pin down a request's details instead of confirming it.
CLARIFIES_REQUEST = r"what (date|day|time)|which (date|day|time)|when would you like|what times? work"
FLAG_REVIEWED = re.compile(
    r"\b(care team|doctor|clinician|nurse|provider)s?\b[^.?!]{0,40}"
    r"\b(has|have|already) (reviewed|seen|looked at|assessed)\b",
    re.IGNORECASE,
)

PROFILE: dict = {}


class Checks:
    """Collects named pass/fail checks for one case."""

    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.items.append((name, bool(ok), detail))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.items)

    def failures(self) -> list[str]:
        return [f"{name}: {detail}" for name, ok, detail in self.items if not ok]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _ref_time(case: dict) -> datetime:
    return datetime.fromisoformat(case.get("reference_time", "2026-03-10T14:30"))


def _observation(spec: dict) -> dict:
    """Fill a partial observation spec to the symptom agent's output shape."""
    symptom = spec["symptom"]
    return {
        "symptom": symptom,
        "keywords": spec.get("keywords", [symptom]),
        "label": spec.get("label", symptom),
        "status": spec.get("status", "present"),
        "onset": spec.get("onset", ""),
        "severity": spec.get("severity", ""),
        "description": spec.get("description", ""),
        "is_patient_report": spec.get("is_patient_report", True),
        "aligns_watch_for_id": spec.get("aligns_watch_for_id", ""),
        "missing_fields": spec.get("missing_fields", []),
    }


def _contains_any(text: str, needles) -> bool:
    if isinstance(needles, str):
        needles = [needles]
    return any(n.lower() in (text or "").lower() for n in needles)


def _match_observation(obs: dict, spec: dict, ref: datetime) -> list[str]:
    """Return the fields of ``spec`` that ``obs`` fails to satisfy."""
    misses = []
    if "symptom_contains" in spec and not _contains_any(obs["symptom"], spec["symptom_contains"]):
        misses.append("symptom")
    if "status" in spec:
        allowed = spec["status"] if isinstance(spec["status"], list) else [spec["status"]]
        if obs["status"] not in allowed:
            misses.append(f"status={obs['status']}")
    for key in ("is_patient_report", "aligns_watch_for_id"):
        if key in spec and obs.get(key) != spec[key]:
            misses.append(f"{key}={obs.get(key)!r}")
    if "severity_contains" in spec and not _contains_any(obs["severity"], spec["severity_contains"]):
        misses.append(f"severity={obs['severity']!r}")
    if "missing_fields" in spec and sorted(obs["missing_fields"]) != sorted(spec["missing_fields"]):
        misses.append(f"missing_fields={obs['missing_fields']}")
    if "missing_includes" in spec and not set(spec["missing_includes"]) <= set(obs["missing_fields"]):
        misses.append(f"missing_fields={obs['missing_fields']}")
    if "onset_days_ago" in spec or "onset_hour_range" in spec:
        onset = obs["onset"]
        if not ISO_MINUTE.match(onset):
            misses.append(f"onset not ISO: {onset!r}")
        else:
            parsed = datetime.fromisoformat(onset)
            if "onset_days_ago" in spec:
                expected = (ref - timedelta(days=spec["onset_days_ago"])).date()
                if parsed.date() != expected:
                    misses.append(f"onset date {parsed.date()} != {expected}")
            if "onset_hour_range" in spec:
                lo, hi = spec["onset_hour_range"]
                if not lo <= parsed.hour <= hi:
                    misses.append(f"onset hour {parsed.hour} not in [{lo},{hi}]")
            if parsed > ref:
                misses.append("onset after message time")
    return misses


def _check_reply_text(checks: Checks, reply: str, case: dict) -> None:
    """Checks that apply to every assistant reply, plus case regexes."""
    checks.add("plain_text", not MARKDOWN.search(reply), f"markdown in: {reply[:120]!r}")
    checks.add("no_completed_action_claim", not COMPLETED_ACTION.search(reply),
               f"claims completion: {COMPLETED_ACTION.search(reply) and COMPLETED_ACTION.search(reply).group(0)!r}")
    checks.add("no_flag_review_claim", not FLAG_REVIEWED.search(reply),
               f"implies review: {FLAG_REVIEWED.search(reply) and FLAG_REVIEWED.search(reply).group(0)!r}")
    checks.add("no_internal_terms", not INTERNAL_TERMS.search(reply),
               f"internal term: {INTERNAL_TERMS.search(reply) and INTERNAL_TERMS.search(reply).group(0)!r}")
    checks.add("no_regreeting", not REGREET.search(reply), f"re-greets: {reply[:80]!r}")
    for pattern in case.get("reply_must_match", []):
        checks.add(f"matches /{pattern}/", re.search(pattern, reply, re.IGNORECASE), reply[:160])
    patterns = list(case.get("reply_must_not_match", []))
    if case.get("no_request_clarification"):
        patterns.append(CLARIFIES_REQUEST)
    for pattern in patterns:
        checks.add(f"avoids /{pattern}/", not re.search(pattern, reply, re.IGNORECASE), reply[:160])


def _new_patient() -> Patient:
    """A patient with the seeded profile but no history, for isolated cases."""
    return Patient.objects.create(profile_data=PROFILE)


class _Rollback(Exception):
    pass


def _in_rollback(fn, case: dict) -> Checks:
    """Run a DB-backed case inside a transaction that is always rolled back."""
    result: dict = {}
    try:
        with transaction.atomic():
            result["checks"] = fn(case)
            raise _Rollback
    except _Rollback:
        pass
    return result["checks"]


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------
def run_intent(case: dict) -> Checks:
    checks = Checks()
    result = agents.classify_intent(case["message"], case.get("chat_state", {}), PROFILE)
    checks.add("intent", result["intent"] == case["expect_intent"],
               f"got {result['intent']}, expected {case['expect_intent']}")
    checks.add("no_confidence", set(result) == {"intent"}, f"keys={sorted(result)}")
    return checks


def run_symptom(case: dict) -> Checks:
    checks = Checks()
    ref = _ref_time(case)
    result = agents.extract_symptoms(
        case["message"], PROFILE, case.get("chat_state", {}),
        previous_assistant_message=case.get("previous_assistant_message", ""),
        reference_time=ref,
    )
    obs = result["observations"]
    summary = ", ".join(
        f"{o['symptom']}={o['status']} onset={o['onset']!r} sev={o['severity']!r} "
        f"wf={o['aligns_watch_for_id']!r} pr={o['is_patient_report']} miss={o['missing_fields']}"
        for o in obs
    ) or "none"
    if "max_observations" in case:
        checks.add("max_observations", len(obs) <= case["max_observations"], summary)
    if "min_observations" in case:
        checks.add("min_observations", len(obs) >= case["min_observations"], summary)
    for spec in case.get("expect_observations", []):
        best = min((_match_observation(o, spec, ref) for o in obs), key=len, default=["no observations"])
        checks.add(f"observation {spec.get('symptom_contains')}", not best,
                   f"{'; '.join(best)} :: {summary}")
    for needle in case.get("forbid_patient_report_of", []):
        bad = [o for o in obs if _contains_any(o["symptom"], needle)
               and o["is_patient_report"] and o["status"] in ("present", "uncertain")]
        checks.add(f"no patient report of {needle}", not bad, summary)
    return checks


def run_flag(case: dict) -> Checks:
    checks = Checks()
    observations = [_observation(o) for o in case["observations"]]
    message_embedding = retrieval.embed_message(case["message"])
    flags = agents.evaluate_flags(observations, PROFILE, [], message_embedding)
    detail = "; ".join(f"{f['watch_for_id']} conf={f['confidence']}" for f in flags) or "none"
    checks.add("flag_decision", bool(flags) == case["expect_flag"],
               f"expected flag={case['expect_flag']}, got {detail}")
    if "expect_watch_for_ids" in case:
        got = sorted(f["watch_for_id"] for f in flags)
        checks.add("watch_for_ids", got == sorted(case["expect_watch_for_ids"]), f"got {got}")
    watch_by_id = {w["id"]: w for w in PROFILE["watch_fors"]}
    by_symptom = {o["symptom"]: o for o in observations}
    for f in flags:
        checks.add("confidence_in_range", 0.0 <= f["confidence"] <= 1.0, str(f["confidence"]))
        # Confidence must be reproducible from embeddings alone (never the LLM).
        obs = by_symptom.get(f["symptom"])
        expected = retrieval.score_watch_for_match(
            obs["keywords"], message_embedding, watch_by_id[f["watch_for_id"]]
        ) if obs else None
        checks.add("confidence_from_embeddings",
                   expected is not None and abs(expected["confidence"] - f["confidence"]) < 1e-6,
                   f"flag={f['confidence']} recomputed={expected and expected['confidence']}")
        checks.add("symptom_links_to_observation", obs is not None, f"symptom={f['symptom']!r}")
        checks.add("has_description", bool(f["description"].strip()), "empty description")
    return checks


def _create_history(patient: Patient, other: Patient | None, items: list[dict]) -> list[int]:
    """Persist prior statuses (each with its own ended session and source message)."""
    now = timezone.now()
    status_ids: list[int] = []
    for item in items:
        owner = other if item.get("patient") == "other" else patient
        session = ChatSession.objects.create(patient=owner, ended_at=now)
        msg = Message.objects.create(
            session=session, role="patient", content=item["message"],
            message_embedding=retrieval.embed_message(item["message"]), sequence_number=1,
        )
        status = PatientStatus.objects.create(
            patient=owner, session=session, source_message=msg,
            date=(now - timedelta(days=item["reported_days_ago"])).date(),
            reported_at=now - timedelta(days=item["reported_days_ago"]),
            clinical_time=(now - timedelta(days=item.get("clinical_days_ago", item["reported_days_ago"]))).date(),
            symptoms=[item["symptom"]], watch_for_id=item.get("watch_for_id", ""),
            symptom_embedding=retrieval.embed_symptom(item["symptom"]),
            message_ids=[msg.pk], symptom_status=item["status"],
        )
        status_ids.append(status.pk)
    return status_ids


def run_retrieval(case: dict) -> Checks:
    checks = Checks()
    patient = _new_patient()
    other = _new_patient()
    status_ids = _create_history(patient, other, case["history"])

    session = ChatSession.objects.create(patient=patient)
    query = Message.objects.create(
        session=session, role="patient", content=case["message"],
        message_embedding=retrieval.embed_message(case["message"]), sequence_number=1,
    )
    timeline = retrieval.build_timeline(
        patient, query, queries=[{"label": case["keywords"][0], "keywords": case["keywords"]}]
    )
    got = [status_ids.index(t["status_id"]) for t in timeline]
    detail = ", ".join(
        f"#{status_ids.index(t['status_id'])} {t['symptom']} conf={t['confidence']}" for t in timeline
    ) or "none"
    if "expect_order" in case:
        checks.add("expected_order", [g for g in got if g in case["expect_order"]] == case["expect_order"],
                   f"got {got} :: {detail}")
    for idx in case.get("expect_exclude", []):
        checks.add(f"excludes #{idx}", idx not in got, detail)
    checks.add("match_reason_valid", all(
        (t["match"] == "semantic" and t["confidence"] > retrieval.MATCH_THRESHOLD)
        or t["match"] in ("watch_for", "history")
        for t in timeline), detail)
    own = set(PatientStatus.objects.filter(patient=patient).values_list("pk", flat=True))
    checks.add("patient_scoped", all(t["status_id"] in own for t in timeline), detail)
    checks.add("source_linked", all(t["source_message_id"] and t["source_excerpt"] for t in timeline), detail)
    return checks


def run_action(case: dict) -> Checks:
    checks = Checks()
    patient = _new_patient()
    session = ChatSession.objects.create(patient=patient, chat_state=case.get("chat_state", {}))
    reply = workflow.run_turn(session, case["message"])
    actions = list(ActionEvent.objects.filter(session=session))
    expected = case["expect_action_type"]
    got = [a.action_type for a in actions]
    checks.add("action_type", got == ([expected] if expected else []), f"got {got}")
    for a in actions:
        checks.add("original_wording", a.request_text == case["message"], repr(a.request_text))
        checks.add("mock_result", a.result.get("status") == "sent_to_care_team"
                   and a.result.get("delivered") is False and a.result.get("booked") is False,
                   str(a.result))
        linked = Message.objects.filter(session=session, role="patient", action=a).exists()
        checks.add("message_linked", linked, "patient message not linked to action")
        checks.add("session_aggregate", a.pk in ChatSession.objects.get(pk=session.pk).action_ids, "")
    if "expect_min_flags" in case:
        n = Flag.objects.filter(patient=patient).count()
        checks.add("flags_raised", n >= case["expect_min_flags"], f"{n} flags")
    _check_reply_text(checks, reply.content, case)
    return checks


def run_responder(case: dict) -> Checks:
    checks = Checks()
    context = {
        "profile": PROFILE,
        "active_workflow": "open_conversation",
        "checkup": {},
        "pending_symptoms": [],
        "recent_messages": [],
        "new_statuses": [],
        "new_flags": [],
        "recorded_actions": [],
        "timeline": [],
        **case["context"],
    }
    reply = agents.compose_reply(context)
    checks.add("non_empty", bool(reply.strip()), "empty reply")
    _check_reply_text(checks, reply, case)
    return checks


def run_summarizer(case: dict) -> Checks:
    checks = Checks()
    patient = _new_patient()
    session = ChatSession.objects.create(
        patient=patient, ended_at=timezone.now(), summary=case.get("prior_summary", ""),
    )
    turns = list(case["messages"])
    filler = case.get("filler_pairs", 0)
    if filler:
        middle = [["patient", f"Just checking in, day {i + 1}, nothing new."] if j == 0
                  else ["assistant", "Thanks for the update. Anything else?"]
                  for i in range(filler) for j in range(2)]
        turns = turns[:2] + middle + turns[2:]
    for seq, (role, content) in enumerate(turns, start=1):
        Message.objects.create(session=session, role=role, content=content, sequence_number=seq)

    wrote = summarizer.summarize_session(session, reason="eval")
    session.refresh_from_db()
    summary, title = session.summary, session.title
    checks.add("summary_written", wrote and bool(summary.strip()), "no summary")
    checks.add("readable_not_json", not summary.lstrip().startswith(("{", "[")), summary[:80])
    checks.add("cites_sources", re.search(r"#\s?\d+", summary), summary[:160])
    words = title.split()
    checks.add("title_length", 0 < len(words) <= 8, repr(title))
    checks.add("checkpoint", session.chat_state.get("last_summarized_sequence") == len(turns),
               str(session.chat_state))
    for needles in case.get("summary_mentions", []):
        checks.add(f"mentions {needles}", _contains_any(summary, needles), summary[:200])
    for needle in case.get("summary_must_not_mention", []):
        checks.add(f"omits {needle}", not _contains_any(summary, needle), summary[:200])
    again = summarizer.summarize_session(session, reason="eval")
    checks.add("idempotent", again is False, "re-summarized with no new messages")
    return checks


def run_workflow(case: dict) -> Checks:
    checks = Checks()
    patient = _new_patient()
    _create_history(patient, None, case.get("history", []))
    # "sessions" runs several chats in turn (each ends before the next
    # starts); "turns" is a single chat. Turn indexes run across all of them.
    replies, timelines, sessions = [], [], []
    for turns in case.get("sessions") or [case["turns"]]:
        if sessions:
            sessions[-1].ended_at = timezone.now()
            sessions[-1].save(update_fields=["ended_at"])
        session = ChatSession.objects.create(patient=patient)
        sessions.append(session)
        for text in turns:
            reply, debug = workflow.run_turn(session, text, return_trace=True)
            replies.append(reply.content)
            node = next(n for n in debug["trace"] if n["node"] == "longitudinal_retrieval")
            timelines.append(node["outputs"]["timeline"])
    session.refresh_from_db()
    state = session.chat_state
    statuses = list(PatientStatus.objects.filter(session__in=sessions))
    flags = list(Flag.objects.filter(patient=patient))
    actions = [a.action_type for a in ActionEvent.objects.filter(session__in=sessions).order_by("created_at")]
    expect = case["expect"]
    status_detail = "; ".join(
        f"{s.label}={s.symptom_status} onset={s.onset!r} sev={s.severity!r} clin={s.clinical_time}"
        for s in statuses
    ) or "none"

    for spec in expect.get("statuses", []):
        matches = [s for s in statuses if _contains_any(s.label, spec["symptom_contains"])]
        if "count" in spec:
            checks.add(f"{spec['symptom_contains']} status count", len(matches) == spec["count"], status_detail)
        s = matches[-1] if matches else None
        checks.add(f"{spec['symptom_contains']} persisted", s is not None, status_detail)
        if s is None:
            continue
        if "status" in spec:
            allowed = spec["status"] if isinstance(spec["status"], list) else [spec["status"]]
            checks.add(f"{s.label} status", s.symptom_status in allowed, status_detail)
        if "severity_contains" in spec:
            checks.add(f"{s.label} severity", _contains_any(s.severity, spec["severity_contains"]), status_detail)
        if spec.get("onset_today"):
            today = timezone.localdate().isoformat()
            checks.add(f"{s.label} onset", bool(ISO_MINUTE.match(s.onset)) and s.onset.startswith(today),
                       status_detail)
            checks.add(f"{s.label} clinical_time", s.clinical_time and s.clinical_time.isoformat() == today,
                       status_detail)
        if "min_message_ids" in spec:
            checks.add(f"{s.label} message_ids", len(s.message_ids) >= spec["min_message_ids"], str(s.message_ids))
    if "min_repeat_reports" in expect:
        repeats = sum(len((f.evidence or {}).get("repeat_reports", [])) for f in flags)
        checks.add("repeat_linked_to_flag", repeats >= expect["min_repeat_reports"],
                   f"{repeats} repeat report(s) linked")
    if "flags_per_session" in expect:
        per = [sum(1 for f in flags if f.patient_status.session_id == ses.pk) for ses in sessions]
        checks.add("flags_per_session", per == expect["flags_per_session"], f"got {per}")
    if "flag_count" in expect:
        checks.add("flag_count", len(flags) == expect["flag_count"],
                   f"{len(flags)} flags: {[f.description[:50] for f in flags]}")
    if "actions" in expect:
        checks.add("actions", actions == expect["actions"], f"got {actions}")
    details = [
        d["detail"]
        for a in ActionEvent.objects.filter(session=session)
        for d in a.follow_up_details
    ]
    if "action_detail_count" in expect:
        checks.add("action_detail_count", len(details) == expect["action_detail_count"], str(details))
    for needles in expect.get("action_details_mention", []):
        checks.add(f"action details mention {needles}", any(_contains_any(d, needles) for d in details),
                   str(details))
    if expect.get("requests_closed"):
        checks.add("requests_closed", not state.get("open_action_ids"), str(state.get("open_action_ids")))
    if expect.get("pending_empty"):
        checks.add("pending_closed", not state.get("pending_symptoms"), str(state.get("pending_symptoms")))
    if "active_workflow" in expect:
        checks.add("active_workflow", state.get("active_workflow") == expect["active_workflow"],
                   str(state.get("active_workflow")))
    checkup = state.get("checkup") or {}
    if "checkup_completed" in expect:
        checks.add("checkup_completed", checkup.get("completed") == expect["checkup_completed"], str(checkup))
    for wid in expect.get("checkup_completed_includes", []):
        checks.add(f"checkup covered {wid}", wid in checkup.get("watch_for_ids_completed", []), str(checkup))
    for reply in replies:
        _check_reply_text(checks, reply, {})
    for turn, statuses_expected in case.get("timeline_by_turn", {}).items():
        got = sorted(f"{e['symptom']}:{e['symptom_status']}" for e in timelines[int(turn)])
        checks.add(f"turn {turn} retrieval", all(x in got for x in statuses_expected),
                   f"got {got}, expected {statuses_expected}")
    for turn, patterns in case.get("reply_must_match_by_turn", {}).items():
        for pattern in patterns:
            checks.add(f"turn {turn} reply matches /{pattern}/",
                       re.search(pattern, replies[int(turn)], re.IGNORECASE), replies[int(turn)][:160])
    for turn, patterns in case.get("reply_must_not_match_by_turn", {}).items():
        for pattern in patterns:
            checks.add(f"turn {turn} reply avoids /{pattern}/",
                       not re.search(pattern, replies[int(turn)], re.IGNORECASE), replies[int(turn)][:160])
    for pattern in case.get("final_reply_must_not_match", []):
        checks.add(f"final reply avoids /{pattern}/", not re.search(pattern, replies[-1], re.IGNORECASE),
                   replies[-1][:160])
    for pattern in case.get("final_reply_must_match", []):
        checks.add(f"final reply matches /{pattern}/", re.search(pattern, replies[-1], re.IGNORECASE),
                   replies[-1][:160])
    return checks


RUNNERS = {
    "intent": (run_intent, False),
    "symptom": (run_symptom, False),
    "retrieval": (run_retrieval, True),
    "flag": (run_flag, False),
    "action": (run_action, True),
    "responder": (run_responder, False),
    "summarizer": (run_summarizer, True),
    "workflow": (run_workflow, True),
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", help="comma-separated stages to run")
    parser.add_argument("--case", help="only cases whose name contains this")
    args = parser.parse_args()

    if not llm.is_live():
        print("Every eval stage calls the live model or embeddings; set OPENAI_API_KEY.")
        return 2

    cases = json.loads(CASES_PATH.read_text())
    stages = set(args.stage.split(",")) if args.stage else set(STAGES)
    unknown = stages - set(STAGES)
    if unknown:
        print(f"Unknown stage(s): {', '.join(sorted(unknown))}. Choose from {', '.join(STAGES)}.")
        return 2
    cases = [c for c in cases if c["stage"] in stages and (not args.case or args.case in c["name"])]
    cases.sort(key=lambda c: STAGES.index(c["stage"]))

    global PROFILE
    old_name = connection.settings_dict["NAME"]
    connection.creation.create_test_db(verbosity=0, autoclobber=True, serialize=False)
    try:
        call_command("seed_initial_data", verbosity=0, stdout=open(os.devnull, "w"))
        PROFILE = Patient.objects.get().profile_data

        tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        print(f"Running {len(cases)} cases against the live model\n")
        stage = None
        for case in cases:
            if case["stage"] != stage:
                stage = case["stage"]
                print(f"== {stage} ==")
            runner, needs_db = RUNNERS[stage]
            try:
                checks = _in_rollback(runner, case) if needs_db else runner(case)
                ok, failures = checks.ok, checks.failures()
            except Exception as exc:  # pragma: no cover - surfaced as a failure
                ok, failures = False, [f"ERROR {type(exc).__name__}: {exc}"]
            tally[stage][0] += ok
            tally[stage][1] += 1
            print(f"[{'PASS' if ok else 'FAIL'}] {case['name']}")
            for failure in failures:
                print(f"         - {failure}")
        print("\nSummary")
        for name in STAGES:
            if name in tally:
                passed, total = tally[name]
                print(f"  {name:<11} {passed}/{total}")
        passed = sum(p for p, _ in tally.values())
        total = sum(t for _, t in tally.values())
        print(f"  {'total':<11} {passed}/{total}")
        return 0 if passed == total else 1
    finally:
        connection.creation.destroy_test_db(old_name, verbosity=0)


if __name__ == "__main__":
    raise SystemExit(main())
