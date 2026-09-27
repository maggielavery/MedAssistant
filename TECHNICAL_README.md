# Maggie Health Assistant

A Django + Django REST Framework + LangGraph prototype of a patient-facing chat
assistant and a care-team dashboard. It is seeded with one patient, Jordan
Ellis (71, CHF with reduced ejection fraction and moderate COPD), plus a prior
session in which Jordan reported mild ankle swelling about 60 days ago and
said it had gone away about 14 days ago, so retrieval has history to find. Everything
patient-specific (conditions, medications, visits, baseline weight, watch-fors
and their guidance) lives in `Patient.profile_data`, set by
`seed_initial_data`. Agent code has no patient-specific logic, so replacing
the profile needs no workflow changes.

Confidence scores are supporting metadata, not clinically validated
thresholds. Appointment and doctor-message requests are mocked: they reach the
care-team dashboard only and are never booked or delivered.

## Quick start

Requirements: Python 3.11+ and an OpenAI API key. No part of the app has an
offline fallback, so seeding, chatting, and summarizing all need the key.

```
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # or paste it into config.py, or use .env
python manage.py migrate
python manage.py seed_initial_data
python manage.py runserver
```

Then open:

- Patient chat: http://127.0.0.1:8000/chat/patients/1/
- Care dashboard: http://127.0.0.1:8000/ops/dashboard/

To summarize idle sessions during development, run this alongside the server:

```
while true; do python manage.py summarize_idle_sessions; sleep 30; done
```

To start over, delete `db.sqlite3` and rerun `migrate` and `seed_initial_data`
(or run `seed_initial_data --reset` to reseed the patient in place).

Configuration (`config.py` reads environment variables first, then its own
defaults; `.env` is loaded if present, see `.env.example`):

| Setting | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | Required for every agent and all embeddings |
| `OPENAI_MODEL` | `gpt-5.4-nano` | Responses API model for every agent |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Message, symptom, and watch-for embeddings |
| `SESSION_IDLE_SECONDS` | `60` | Idle threshold for the summarizer |
| `RECENT_MESSAGE_WINDOW` | `12` | Recent messages given to the responder |
| `DATABASE_URL` | SQLite `db.sqlite3` | Optional; models stay PostgreSQL-compatible |

`TIME_ZONE` in `continuum/settings.py` is `UTC`. Relative onsets such as "this
morning" are resolved in that zone, so set it to your local zone if chats
happen late in the day.

## Features

**Patient chat.** An LLM-style chat with a sidebar of conversations by
generated title, New chat and End chat buttons, and an assistant greeting at
the start of each session. Starting a new chat ends the previous one, and only
the latest un-ended session is writable; earlier sessions are read-only. The
patient can talk openly, ask for a routine check-in, request or change an
appointment, or ask for a message to be relayed to the doctor. While the
assistant works, an animated three-dot indicator shows in its place and the
composer is disabled. If a turn fails, the patient's message is removed, the
text is put back in the composer to resend, and the error is shown.

A debug pane in the chat page shows each workflow stage's inputs and outputs
and the final responder prompt. It is marked `[TEMP DEBUG]` in `chat.html`,
`views.py`, and `workflow.py` for removal.

**Care dashboard.** Patient-scoped tabs, refreshed every 10 seconds (paused
while a dropdown or input has focus). The header shows the patient's name,
sex, age, and patient number with a placeholder photo. Alerts, Appointments,
and Doctor messages show a red count of items that are new since that pane was
last opened (a new flag or request, or new details added to a request). The
count clears only when the pane is opened; what has been seen is remembered
per patient in the browser (`localStorage`).

- Overview: a left profile column (placeholder photo, profile, care guidance,
  conditions, medications, recent visit, upcoming appointment) and a symptom
  timeline: the latest status of each symptom (grouped by watch-for, else
  label) as colored chips, then recent reports by day, each with its status,
  onset, severity, still-missing details, and description.
- 🔴 Alerts: flags with description, source message and quote, watch-for,
  confidence with its two similarity components, still-missing details, an
  editable status (NEW, OPEN, ACKNOWLEDGED, RESOLVED, DISMISSED), and when it
  was raised.
- 📅 Appointments: the recent visit and upcoming appointment at the top, then
  appointment requests and changes (the patient's original request, details
  added later in the chat, and the mock result).
- 📌 Doctor messages: relay requests with the full patient message, added
  details, and the mock result.

## Agent workflow

Each patient message runs through explicit LangGraph stages in
`assistant_app/workflow.py`. The graph state is a plain dict, so each stage can
be evaluated on its own.

```
persist raw Message (with embedding)
  -> load profile + recent messages + chat_state
  -> intent agent
  -> symptom agent (+ pending follow-up merge, checkup progress)
  -> longitudinal retrieval
  -> flag agent + action-update agent
  -> persist statuses, flags, actions
  -> responder agent
  -> persist assistant Message
```

All LLM calls use temperature 0 through `llm.py`. There are no offline
fallbacks: a missing key, a failed request, unparseable JSON, or a malformed
result (an unknown intent label, a missing observations/flags/updates list)
raises `llm.LLMUnavailable` instead of continuing on heuristic output.

- A chat turn runs in one transaction, so a failure keeps nothing, not even
  the patient message; the chat API returns 503 and the patient can resend.
- New chat embeds the greeting before changing anything, so a failure leaves
  the previous session open (503).
- Ending a chat still ends it; a failed summary is retried by
  `summarize_idle_sessions`, which skips and reports sessions it can't
  summarize.
- `seed_initial_data` is all-or-nothing and stops with a clear error.

### Intent agent

Classifies the message as `routine_checkup`, `appointment_request`,
`appointment_change`, `contact_doctor`, or `general_chat` by meaning, not
keywords. A symptom report, medical question, greeting, or vague "go over"
is `general_chat`. A code guard downgrades `routine_checkup` to
`general_chat` unless the message explicitly asks for a check-in (for example
"Can we do my routine check-in?"), so ordinary chat never switches the
workflow. An appointment request during a checkup keeps its appointment
intent. The intent is stored on `Message`; it has no confidence score.

### Symptom agent

Extracts every clinically relevant observation the patient actually states,
including ones outside the watch-fors, and returns nothing for greetings,
thanks, scheduling, or questions. Each observation has a concise `symptom`,
retrieval `keywords`, `status` (present, absent, historical, uncertain,
resolved), onset, severity, description, whether it is the patient's own
report (not hypothetical, family, or someone else), and the matching
watch-for ID, which is dropped if it is not a real profile ID.

It also receives the assistant's previous message (context only, never mined
for symptoms) so a short reply like "6" after "rate it from 1 to 10" becomes
severity `6/10`, and the message time so onset becomes an approximate ISO
datetime (`YYYY-MM-DDTHH:MM`): "this morning" is that date at 08:00, "last
night" the prior date at 21:00, "yesterday" or "two days ago" that date at
12:00. Onset is clamped to no later than the message time, and its date is
stored as `PatientStatus.clinical_time`. Weight gain goes through this same
path; the "more than 3 lbs in 2 days" rule is watch-for guidance in seed data,
not code.

**Follow-ups.** `missing_fields` is computed in code as whichever of `onset`
and `severity` is still empty for a present or uncertain observation; the
model's own list is ignored because free-text items could never be cleared.
An incomplete observation waits in `chat_state.pending_symptoms` while the
responder asks about one missing detail at a time. Follow-up answers are
merged into the same record (description appended, every contributing message
added to `message_ids`). After two follow-up answers the observation is saved
anyway, with "Not provided after follow-up: <fields>" in its description, so
an "I don't know" can't loop. A reply that doesn't address the pending
question at all (a non-answer or a change of subject) also counts toward
those two, except an appointment or doctor-message request, which is treated
as an interruption. A complete observation becomes one
`PatientStatus` row. A later report creates a new row rather than overwriting
history.

**Checkup.** A routine check-in walks the profile's watch-fors in order. The
symptom agent is told which item was just asked, so a reply about it,
including a denial like "no, my weight has been steady", is returned as an
observation aligned to that item (absent for a denial). A present, absent,
uncertain, or resolved answer about the current item marks it done and moves
to the next. Interruptions such as an appointment request are handled
without losing progress, and the checkup ends when every item is covered.

### Longitudinal retrieval

Runs one search per topic in play: each observation this turn, each pending
symptom, and, during a routine check-in, the item the assistant is about to
ask (so its history is available before the patient mentions it). With no
topic (for example "I just want to chat") nothing is searched. Each search
covers only the current patient's `PatientStatus` rows:

```
symptom_similarity = cosine(embed(symptom + keywords), status.symptom_embedding)
message_similarity = cosine(query context embedding, status source message embedding)
confidence         = 0.7 * symptom_similarity + 0.3 * message_similarity
```

Searches are scored separately so one symptom's words never dilute another's.
The query context is the symptom's own source message (where it was first
reported, so a follow-up like "since yesterday, mild" still matches its
history); for a check-in item it is the item's "label. guidance" text. A status
keeps its best score across searches.

Matching and history, in order:

1. `semantic`: `confidence > 0.6`.
2. `watch_for`: the row has the same profile watch-for as the search (stored
   on `PatientStatus.watch_for_id` when the observation is saved), whatever
   its score.
3. The six strongest matches are kept, then `history` expansion adds the rest
   of each matched symptom's history: rows with the same watch-for, or the same
   label if it has none (at most eight most recent per symptom).

Scores rank results but never drop a row that belongs to a matched symptom,
so a later "resolved" report is returned even when it is worded nothing like
the new complaint ("my legs are back to normal"). Results are ordered by
clinical date and then report time, so an old event mentioned recently sorts
by when it happened. The symptom agent's keywords are limited to symptom terms
(no numbers or words like "steady"). Each timeline entry carries the source
message ID and excerpt, report and clinical times, status, watch-for,
`message_ids`, the scores, the matching search, and why it matched. Session
summaries are never used as clinical facts.

`retrieval_eval/` measures this on a labeled benchmark. At the 0.6 threshold,
the spec's scoring alone found 77% of history rows and the current status for
60% of queries; with the watch-for match and history expansion both reached
100%, with no irrelevant rows returned. See
[retrieval_eval/README.md](retrieval_eval/README.md) for the dataset, sweeps,
and caveats.

### Flag agent

Only present or uncertain patient reports aligned to a real watch-for are
candidates; absent and historical observations are never flagged. The LLM
decides which candidates meet their watch-for guidance (thresholds
interpreted semantically) and writes the description. It does not score them.
Confidence is computed from embeddings with the retrieval weighting:

```
symptom_similarity = cosine(embed(keywords), embed(watch-for label))
message_similarity = cosine(message embedding, embed("label. guidance"))
confidence         = 0.7 * symptom_similarity + 0.3 * message_similarity
```

There is no cutoff on flag confidence; the guidance check is the decision. A
concerning observation is flagged even while details are missing: its status
row is saved immediately (with `missing_fields` set) so the flag reaches the
dashboard, and later follow-up answers complete that same row. A report in a new chat session always gets its own flag in the Alerts pane,
whatever happened to earlier flags for that concern. Within one session, while
an active (NEW, OPEN, ACKNOWLEDGED) flag exists for the same concern (same
watch-for, else same label), no second flag is created. Instead the new report
is linked to the session's active flag: it is appended to
`Flag.evidence.repeat_reports` (message, status, session, quote, time) and the
status links to the flag. The Alerts pane shows "Reported again N×" with the
latest quote, and each repeat brings back the Alerts badge. Follow-up answers
that complete an already-linked report are not counted again.

### Action agent

`appointment_request`, `appointment_change`, and `contact_doctor` intents
create an `ActionEvent` holding the patient's original wording, with result
`{"status": "sent_to_care_team", "booked": false, "delivered": false}`.
Flags never create actions.

A new request is open for details (`chat_state.open_action_ids`). While a
request is open, one LLM call decides whether the next message adds something
the care team should see with it, such as a preferred date or time, a reason,
or onset and severity for a symptom in a doctor message. Each detail is
appended to `ActionEvent.follow_up_details` without adding facts, and the
request stays open. The first reply that adds nothing closes it for good:
"that's everything", thanks, or moving on (a question, a new symptom, a
check-in answer). Closed requests are never considered again, so later chat
can't keep appending to a doctor message. A repeat of the same request type
that updates an open request ("actually, make that Thursday") does not create
a duplicate; the same type after the request closed is a new request.

### Responder agent

Builds a bounded prompt from the patient profile, the last 12 messages,
`chat_state` (workflow, checkup progress, pending questions), this turn's
observations, flags, and actions, and the retrieved timeline. Priorities, in
order:

1. A request sent this turn is confirmed first: "your appointment change
   request has been sent to the care team", and someone will contact the
   patient about availability or next steps. No questions about the request's
   date or details, and never "booked", "rescheduled", or "sent to the doctor".
   Unless a symptom question must come first, the reply ends by asking
   whether there is anything else to add to the request, or if that's
   everything.
2. A detail added to an open request is acknowledged without re-confirming,
   followed by the same "anything else?" question. When the patient is done
   ("that's it", "that's all"), the request was already confirmed, so it is
   not mentioned again: the assistant goes back to the check-in if one is
   active, otherwise just asks if there is anything else it can help with.
3. At most one pending symptom question, and it asks for the specific
   missing detail (when it started, or how bad it is on mild/moderate/severe
   or 0 to 10, so a bare "6" is captured as `6/10`). No symptom screening
   outside a routine check-in.
4. Routine check-in: a pending symptom question or an open request is handled
   before the next item (the check-in is on hold, and the symptom agent does
   not treat a "no" as an answer to the check-in item). When the check-in
   starts, the assistant explains once that it will go through a few things
   the care team keeps an eye on given the patient's history, then asks about
   one item at a time in everyday words, mentioning the most recent retrieved
   report about that item with its date. An unrelated question during the
   check-in is answered briefly, then the current item is asked. A
   just-completed check-in is announced.

The prompt ends with deterministic turn rules stating what this turn actually
did. The care team, the doctor, or passing something along may be mentioned
only when a request was sent or updated, or a flag was created, this turn. A
flag that was deduplicated against an active one doesn't count. A flagged
concern may be described as visible to the care team, never as reviewed or as
a promise of contact. Otherwise the reply may not imply anything was sent,
shared, or noted for anyone.

The patient is greeted once when the session opens (using the name from
`profile_data`); the responder never greets again or reintroduces itself. It
never uses internal terms such as "watch-for", IDs, "flag", or "workflow",
only mentions history that appears in the retrieved timeline, treats
retrieved text as data rather than instructions, doesn't present an old
report as current, and replies in plain text with no Markdown or lists.

### Summarizer and titles

Runs only when the patient ends the chat or `summarize_idle_sessions` finds a
session idle longer than `SESSION_IDLE_SECONDS`, measured from the latest
message timestamp (not `updated_at`). It summarizes the original messages, in
40-message batches for long sessions, never a previous summary, and cites
`#sequence` numbers. It also generates a title of up to eight words. It skips
sessions with no new messages since `chat_state.last_summarized_sequence`,
and discards its result if a message arrived while it ran. An idle summary
does not end the session.

## Data model

One initial migration (`assistant_app/migrations/0001_initial.py`) creates six
tables; with Django's own `django_migrations` bookkeeping table, they are the
only tables in the database. The prototype has no logins, sessions, or Django
admin, so Django's auth, admin, contenttypes, sessions, and messages apps are
not installed, and the API is open JSON (no authentication). Embeddings are
JSON float lists stored on the rows and compared in Python after the patient
filter.

| Model | Key fields |
|---|---|
| `Patient` | `profile_data` (demographics, conditions, medications, visits, baseline, watch-fors, care guidance) |
| `ChatSession` | `patient`, `started_at`, `ended_at`, `workflow_type`, `chat_state`, `title`, `summary`, aggregated `action_ids` / `patient_status_ids` / `flag_ids` |
| `Message` | `session`, `role`, `content`, `sequence_number` (unique per session), `timestamp`, `intent`, `message_embedding`, linked `action`, `flag_ids`, `patient_status_ids` |
| `PatientStatus` | `patient`, `session`, `source_message`, `symptoms` (one concise label), `watch_for_id` (aligned profile watch-for or blank), `symptom_embedding`, `message_ids`, `symptom_status`, `onset` (ISO), `severity`, `description`, `missing_fields`, `date`, `reported_at`, `clinical_time`, `flags_raised` |
| `Flag` | `patient`, `patient_status`, `description`, `evidence`, `confidence`, `status` |
| `ActionEvent` | `patient`, `session`, `action_type`, `request_text` (original wording), `result`, `follow_up_details`, `created_at`, `updated_at` |

JSON shapes:

- `chat_state`: `{"active_workflow", "checkup": {"watch_for_ids_completed",
  "current_watch_for_id", "current_watch_for_label", "completed"},
  "pending_symptoms": [{symptom, keywords, status, onset, severity,
  description, aligns_watch_for_id, message_ids, missing_fields, status_id,
  follow_ups_asked, question}], "open_action_ids", "last_summarized_sequence"}`
- `Flag.evidence`: `{source_message_id, message_ids, quote, watch_for_id,
  symptom_similarity, message_similarity, missing_fields, reported_at,
  repeat_reports: [{message_id, status_id, session_id, quote, reported_at}]}`
- `ActionEvent.follow_up_details`: `[{message_id, detail, quote, added_at}]`

## API

```
GET   /api/patients/<patient_id>/sessions/            # title, summary, ended_at, writable
POST  /api/patients/<patient_id>/sessions/new/        # ends the previous session
GET   /api/sessions/<session_id>/messages/
POST  /api/sessions/<session_id>/chat/                # rejected for read-only sessions
POST  /api/sessions/<session_id>/end/                 # sets ended_at, summarizes

GET   /api/admin/patients/
GET   /api/admin/patients/<patient_id>/dashboard/
GET   /api/admin/database-tables/?patient_id=<id>[&preview=1]
GET   /api/admin/alerts/<flag|appointment|doctor_message>/<id>/
PATCH /api/admin/flags/<flag_id>/                     # {"status": ...}
PATCH /api/admin/actions/<action_id>/                 # {"status" | "note"} in result
```

`database-tables` is an API-only view (not shown in the dashboard) that returns
every stored column per table, with foreign keys as `<name>_id`; `preview=1`
shortens embeddings in the response only.

## Evaluation

```
python eval/run_eval.py                        # every case
python eval/run_eval.py --stage flag,symptom   # selected stages
python eval/run_eval.py --case weight          # cases whose name contains this
```

`eval/synthetic_cases.json` has cases for every stage, scored by deterministic
checks (not an LLM judge), so a failure names the exact expectation missed:

| Stage | Checks |
|---|---|
| intent | Labels, explicit-only checkups, appointments during a checkup, no confidence |
| symptom | Status, watch-for alignment (including check-in denials), reporter, resolved/historical, computed missing fields, ISO onset, short answers read from context |
| retrieval | The 0.6 threshold, patient scoping, source links, clinical-time ordering |
| flag | Decision against guidance (including below-threshold weight gain); confidence reproducible from embeddings |
| action | Type, original wording, mock result, message linkage, care-team confirmation without date questions |
| responder | Confirmation before symptom questions, added details, no booking/delivery/review claims, checkup completion, prompt injection, plain text, no internal terms or re-greeting |
| summarizer | Source citations, title length, ignores a stale summary, batching, idempotency |
| workflow | Multi-turn follow-ups, checkup interruptions, check-in retrieval for the next item and for follow-up answers, pending question before the next item, duplicate flags within a session vs. new flags across sessions, request updates vs. separate requests, requests closing on "that's everything" or a change of topic |

Every stage needs `OPENAI_API_KEY`. Database-backed stages run in a throwaway
test database that is migrated and seeded each run, with each case rolled
back, so `db.sqlite3` is never touched. The eval profile comes from the seed
command.

For retrieval specifically, `python retrieval_eval/run_experiment.py` runs a
labeled benchmark comparing the spec's scoring with the shipped version, and
sweeps the threshold and weights (charts in `retrieval_eval/results/`).

## Project layout

```
config.py                         OpenAI settings (env first, then defaults)
continuum/                        Django settings and root URLs
assistant_app/
  models.py                       schema above
  workflow.py                     LangGraph stages, persistence, chat_state logic
  agents.py                       intent, symptom, flag, action-update, responder
  retrieval.py                    embeddings, patient-scoped search, shared scoring
  summarizer.py                   whole-session summaries and titles
  llm.py                          OpenAI wrapper; raises LLMUnavailable
  views.py, serializers.py        API and page handlers
  templates/assistant_app/        chat.html, dashboard.html
  management/commands/            seed_initial_data, summarize_idle_sessions
eval/
  run_eval.py, synthetic_cases.json
retrieval_eval/                   labeled retrieval benchmark, sweeps, and charts
  run_experiment.py, dataset.json, results/
```
