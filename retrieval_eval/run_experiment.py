"""Longitudinal-retrieval experiment.

Runs the real ``assistant_app.retrieval.search_statuses`` over the labeled
benchmark in ``dataset.json`` and compares four variants:

    spec_baseline   score > 0.6 only (0.7 symptom + 0.3 message)
    +watch_for      also match rows with the query's exact watch-for
    +history        also expand each matched symptom to its full history
    full            both (the shipped configuration)

It then sweeps the match threshold and the symptom/message weighting, and
plots the score distributions of relevant vs. irrelevant rows.

Embeddings come from the configured OpenAI embedding model and are cached in
``embedding_cache.json`` so re-runs and sweeps are free and reproducible.
Everything runs in a throwaway test database; db.sqlite3 is never touched.

    python retrieval_eval/run_experiment.py
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from datetime import timedelta
from pathlib import Path

import django

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "continuum.settings")
django.setup()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from django.db import connection  # noqa: E402
from django.utils import timezone  # noqa: E402

from assistant_app import llm, retrieval  # noqa: E402
from assistant_app.models import ChatSession, Message, Patient, PatientStatus  # noqa: E402

DATASET = json.loads((HERE / "dataset.json").read_text())
CACHE_PATH = HERE / "embedding_cache.json"
RESULTS = HERE / "results"
VARIANTS = {
    "spec_baseline": {"use_watch_for": False, "expand_history": False},
    "+watch_for": {"use_watch_for": True, "expand_history": False},
    "+history": {"use_watch_for": False, "expand_history": True},
    "full": {"use_watch_for": True, "expand_history": True},
}
THRESHOLDS = [round(0.40 + 0.025 * i, 3) for i in range(17)]  # 0.40 .. 0.80
SYMPTOM_WEIGHTS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ---------------------------------------------------------------------------
# Embedding cache (wraps llm.embed so the production code path is unchanged)
# ---------------------------------------------------------------------------
_cache: dict[str, list[float]] = (
    json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
)
_real_embed = llm.embed


def _cached_embed(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        return []
    key = hashlib.sha1(f"{llm.settings.OPENAI_EMBEDDING_MODEL}|{text}".encode()).hexdigest()
    if key not in _cache:
        _cache[key] = [round(x, 7) for x in _real_embed(text)]
    return _cache[key]


llm.embed = _cached_embed


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
def load_fixture() -> tuple[Patient, dict[int, dict], list[dict]]:
    """Create both patients' history and one query message per case."""
    now = timezone.now()
    patient = Patient.objects.create(profile_data={"demographics": {"name": "Benchmark"}})
    other = Patient.objects.create(profile_data={"demographics": {"name": "Other"}})
    rows: dict[int, dict] = {}
    for item in DATASET["history"]:
        owner = other if item.get("patient") == "other" else patient
        session = ChatSession.objects.create(patient=owner, ended_at=now)
        msg = Message.objects.create(
            session=session, role="patient", content=item["message"],
            message_embedding=retrieval.embed_message(item["message"]), sequence_number=1,
        )
        reported = now - timedelta(days=item["reported_days_ago"])
        status = PatientStatus.objects.create(
            patient=owner, session=session, source_message=msg,
            date=reported.date(), reported_at=reported,
            clinical_time=(now - timedelta(days=item.get("clinical_days_ago", item["reported_days_ago"]))).date(),
            symptoms=[item["label"]], watch_for_id=item["watch_for_id"],
            symptom_embedding=retrieval.embed_symptom(item["label"]),
            message_ids=[msg.pk], symptom_status=item["status"],
        )
        rows[status.pk] = {**item, "owner": "other" if owner == other else "self"}

    cases = []
    query_session = ChatSession.objects.create(patient=patient)
    for seq, case in enumerate(DATASET["queries"], start=1):
        msg = Message.objects.create(
            session=query_session, role="patient", content=case["message"],
            message_embedding=retrieval.embed_message(case["message"]), sequence_number=seq,
        )
        queries = [
            {"label": q["label"], "keywords": [q["label"], *q.get("keywords", [])],
             "watch_for_id": q.get("watch_for_id", ""), "context_text": q.get("context_text")}
            for q in case["queries"]
        ]
        cases.append({**case, "message_obj": msg, "built_queries": queries})
    return patient, rows, cases


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _latest_key(family: str) -> str:
    items = [h for h in DATASET["history"] if h["family"] == family]
    latest = min(items, key=lambda h: (h.get("clinical_days_ago", h["reported_days_ago"]), h["reported_days_ago"]))
    return latest["key"]


def evaluate(patient, rows, cases, variant: dict) -> dict:
    per_case = []
    for case in cases:
        hits = retrieval.search_statuses(
            patient, case["message_obj"], queries=case["built_queries"], **variant
        )
        got = [rows[h["status"].pk] for h in hits]
        got_keys = {g["key"] for g in got}
        expect = set(case.get("expect", []))
        ok_families = expect | set(case.get("acceptable", []))
        relevant = [g for g in got if g["family"] in ok_families]
        family_recall = [
            len(got_keys & {h["key"] for h in DATASET["history"] if h["family"] == f})
            / sum(1 for h in DATASET["history"] if h["family"] == f)
            for f in expect
        ]
        per_case.append({
            "id": case["id"],
            "returned": [f"{g['key']}({g['family']})" for g in got],
            "has_history": bool(expect),
            "recall": statistics.mean(family_recall) if family_recall else None,
            "latest_hit": all(_latest_key(f) in got_keys for f in expect) if expect else None,
            "precision": (len(relevant) / len(got)) if got else None,
            "false_hit": (not expect) and any(g["family"] not in ok_families for g in got),
            "leak": any(g["owner"] == "other" for g in got),
        })

    def mean(key, cond=lambda c: True):
        values = [c[key] for c in per_case if cond(c) and c[key] is not None]
        return round(statistics.mean(values), 3) if values else None

    history_cases = lambda c: c["has_history"]  # noqa: E731
    return {
        "family_recall": mean("recall", history_cases),
        "latest_status_hit": mean("latest_hit", history_cases),
        "precision": mean("precision"),
        "no_history_false_hit_rate": mean("false_hit", lambda c: not c["has_history"]),
        "other_patient_leaks": sum(c["leak"] for c in per_case),
        "cases": per_case,
    }


def pair_scores(patient, rows, cases) -> tuple[list[float], list[float]]:
    """Best score of every (case, own-patient row) pair, split by relevance."""
    relevant, irrelevant = [], []
    statuses = list(PatientStatus.objects.filter(patient=patient).select_related("source_message"))
    for case in cases:
        expect = set(case.get("expect", []))
        acceptable = set(case.get("acceptable", []))
        for status in statuses:
            family = rows[status.pk]["family"]
            if family in acceptable:
                continue
            best = 0.0
            for q in case["built_queries"]:
                k = retrieval.embed_symptom(" ".join(q["keywords"]))
                ctx = (list(retrieval._embed_profile_text(q["context_text"])) if q.get("context_text")
                       else case["message_obj"].message_embedding)
                best = max(best, retrieval.weighted_score(
                    llm.cosine_similarity(k, status.symptom_embedding),
                    llm.cosine_similarity(ctx, status.source_message.message_embedding),
                ))
            (relevant if family in expect else irrelevant).append(best)
    return relevant, irrelevant


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run() -> None:
    RESULTS.mkdir(exist_ok=True)
    patient, rows, cases = load_fixture()
    original = (retrieval.MATCH_THRESHOLD, retrieval.SYMPTOM_WEIGHT, retrieval.MESSAGE_WEIGHT)

    comparison = {name: evaluate(patient, rows, cases, v) for name, v in VARIANTS.items()}

    threshold_sweep = {name: [] for name in ("spec_baseline", "full")}
    for t in THRESHOLDS:
        retrieval.MATCH_THRESHOLD = t
        for name in threshold_sweep:
            m = evaluate(patient, rows, cases, VARIANTS[name])
            threshold_sweep[name].append({"threshold": t, **{k: v for k, v in m.items() if k != "cases"}})
    retrieval.MATCH_THRESHOLD = original[0]

    weight_sweep = []
    for w in SYMPTOM_WEIGHTS:
        retrieval.SYMPTOM_WEIGHT, retrieval.MESSAGE_WEIGHT = w, round(1 - w, 2)
        m = evaluate(patient, rows, cases, VARIANTS["spec_baseline"])
        weight_sweep.append({"symptom_weight": w, **{k: v for k, v in m.items() if k != "cases"}})
    retrieval.MATCH_THRESHOLD, retrieval.SYMPTOM_WEIGHT, retrieval.MESSAGE_WEIGHT = original

    relevant, irrelevant = pair_scores(patient, rows, cases)

    (RESULTS / "metrics.json").write_text(json.dumps({
        "comparison": comparison, "threshold_sweep": threshold_sweep,
        "weight_sweep": weight_sweep,
        "score_distribution": {"relevant": relevant, "irrelevant": irrelevant},
    }, indent=2))
    _plots(comparison, threshold_sweep, weight_sweep, relevant, irrelevant)
    _summary(comparison, threshold_sweep, weight_sweep, relevant, irrelevant)
    print((RESULTS / "summary.md").read_text())


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
METRICS = [
    ("family_recall", "Family recall"),
    ("latest_status_hit", "Latest-status hit"),
    ("precision", "Precision"),
    ("no_history_false_hit_rate", "False-hit rate (no history)"),
]
COLORS = {"family_recall": "#2a6fdb", "latest_status_hit": "#1b9e77",
          "precision": "#d95f02", "no_history_false_hit_rate": "#7570b3"}


def _plots(comparison, threshold_sweep, weight_sweep, relevant, irrelevant) -> None:
    # 1. Variant comparison
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = list(comparison)
    width = 0.2
    for i, (key, label) in enumerate(METRICS):
        values = [comparison[n][key] or 0 for n in names]
        bars = ax.bar([x + (i - 1.5) * width for x in range(len(names))], values, width,
                      label=label, color=COLORS[key])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, 1.12)
    ax.set_title("Retrieval variants at threshold 0.6")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "variant_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Threshold sweep
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, threshold_sweep):
        xs = [r["threshold"] for r in threshold_sweep[name]]
        for key, label in METRICS:
            ax.plot(xs, [r[key] if r[key] is not None else float("nan") for r in threshold_sweep[name]],
                    label=label, color=COLORS[key], marker="o", markersize=3)
        ax.axvline(0.6, color="#999", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("match threshold")
        ax.set_ylim(-0.02, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Threshold sweep (dashed line = shipped 0.6)")
    fig.tight_layout()
    fig.savefig(RESULTS / "threshold_sweep.png", dpi=150)
    plt.close(fig)

    # 3. Weight sweep (spec baseline)
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r["symptom_weight"] for r in weight_sweep]
    for key, label in METRICS:
        ax.plot(xs, [r[key] if r[key] is not None else float("nan") for r in weight_sweep],
                label=label, color=COLORS[key], marker="o", markersize=3)
    ax.axvline(0.7, color="#999", linestyle="--", linewidth=1)
    ax.set_xlabel("symptom weight (message weight = 1 - w)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Weight sweep, spec baseline at threshold 0.6")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "weight_sweep.png", dpi=150)
    plt.close(fig)

    # 4. Score distributions
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = [i / 40 for i in range(41)]
    ax.hist(irrelevant, bins=bins, alpha=0.6, label=f"irrelevant rows (n={len(irrelevant)})", color="#999")
    ax.hist(relevant, bins=bins, alpha=0.8, label=f"relevant rows (n={len(relevant)})", color="#2a6fdb")
    ax.axvline(0.6, color="#d95f02", linestyle="--", linewidth=1.2, label="threshold 0.6")
    ax.set_xlabel("combined score (0.7 symptom + 0.3 message)")
    ax.set_ylabel("query-row pairs")
    ax.set_title("Score separation: relevant vs. irrelevant history rows")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "score_distribution.png", dpi=150)
    plt.close(fig)


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


def _summary(comparison, threshold_sweep, weight_sweep, relevant, irrelevant) -> None:
    lines = ["# Retrieval experiment results", "",
             f"{len(DATASET['queries'])} queries over {len(DATASET['history'])} history rows "
             f"({sum(1 for q in DATASET['queries'] if q['expect'])} with history, "
             f"{sum(1 for q in DATASET['queries'] if not q['expect'])} without).", "",
             "## Variants at threshold 0.6", "",
             "| Variant | " + " | ".join(label for _, label in METRICS) + " | Other-patient leaks |",
             "|---|" + "---|" * (len(METRICS) + 1)]
    for name, m in comparison.items():
        lines.append(f"| {name} | " + " | ".join(_fmt(m[k]) for k, _ in METRICS)
                     + f" | {m['other_patient_leaks']} |")

    lines += ["", "## Threshold sweep", "",
              "| Threshold | Baseline recall | Baseline latest | Baseline false-hit | "
              "Full recall | Full latest | Full precision | Full false-hit |", "|---|---|---|---|---|---|---|---|"]
    for b, f in zip(threshold_sweep["spec_baseline"], threshold_sweep["full"]):
        lines.append(f"| {b['threshold']:.3f} | {_fmt(b['family_recall'])} | {_fmt(b['latest_status_hit'])} | "
                     f"{_fmt(b['no_history_false_hit_rate'])} | {_fmt(f['family_recall'])} | "
                     f"{_fmt(f['latest_status_hit'])} | {_fmt(f['precision'])} | {_fmt(f['no_history_false_hit_rate'])} |")

    lines += ["", "## Weight sweep (spec baseline, threshold 0.6)", "",
              "| Symptom weight | " + " | ".join(label for _, label in METRICS) + " |",
              "|---|" + "---|" * len(METRICS)]
    for r in weight_sweep:
        lines.append(f"| {r['symptom_weight']:.1f} | " + " | ".join(_fmt(r[k]) for k, _ in METRICS) + " |")

    def pct(values, cut):
        return sum(v > cut for v in values) / len(values)

    lines += ["", "## Score separation", "",
              f"- Relevant pairs: n={len(relevant)}, median {statistics.median(relevant):.3f}, "
              f"{pct(relevant, 0.6):.0%} above 0.6.",
              f"- Irrelevant pairs: n={len(irrelevant)}, median {statistics.median(irrelevant):.3f}, "
              f"max {max(irrelevant):.3f}, {pct(irrelevant, 0.6):.0%} above 0.6.",
              "", "## Per-query results (full variant)", "",
              "| Query | Recall | Latest hit | Returned |", "|---|---|---|---|"]
    for c in comparison["full"]["cases"]:
        lines.append(f"| {c['id']} | {_fmt(c['recall'])} | {_fmt(c['latest_hit'])} | "
                     f"{', '.join(c['returned']) or 'none'} |")
    lines += ["", "## Per-query results (spec baseline)", "",
              "| Query | Recall | Latest hit | Returned |", "|---|---|---|---|"]
    for c in comparison["spec_baseline"]["cases"]:
        lines.append(f"| {c['id']} | {_fmt(c['recall'])} | {_fmt(c['latest_hit'])} | "
                     f"{', '.join(c['returned']) or 'none'} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    if not llm.is_live():
        print("The experiment embeds text with the live model; set OPENAI_API_KEY.")
        return 2
    old_name = connection.settings_dict["NAME"]
    connection.creation.create_test_db(verbosity=0, autoclobber=True, serialize=False)
    try:
        run()
    finally:
        connection.creation.destroy_test_db(old_name, verbosity=0)
        CACHE_PATH.write_text(json.dumps(_cache))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
