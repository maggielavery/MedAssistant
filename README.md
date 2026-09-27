# Takehome Health Assistant

A prototype of a patient-facing chat assistant and a care-team dashboard, built with Django, Django REST Framework, LangGraph, and OpenAI. The full technical write-up (agent stages, retrieval math, data model, API, evals) is in [TECHNICAL.md].

## Setup

Requires Python 3.11+ and an OpenAI API key.

```
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # or copy .env.example to .env
python manage.py migrate
python manage.py seed_initial_data
python manage.py runserver
```

- Patient chat: http://127.0.0.1:8000/chat/patients/1/
- Care dashboard: http://127.0.0.1:8000/ops/dashboard/

To summarize idle sessions during development, run alongside the server:

```
while true; do python manage.py summarize_idle_sessions; sleep 30; done
```

To reset, delete `db.sqlite3` and rerun `migrate` and `seed_initial_data`, or run `seed_initial_data --reset`.

Evals: `python eval/run_eval.py` (all stages) and `python retrieval_eval/run_experiment.py` (retrieval benchmark and threshold sweep).

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | **Yes** | none | All agents, embeddings, seeding, summaries, and evals |
| `OPENAI_MODEL` | No | `gpt-5.4-nano` | Model for every agent (Responses API) |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Symptom, message, and watch-for embeddings |
| `SESSION_IDLE_SECONDS` | No | `60` | Idle time before a session is summarized |
| `RECENT_MESSAGE_WINDOW` | No | `12` | Recent messages passed to the responder |
| `DATABASE_URL` | No | SQLite `db.sqlite3` | Optional alternative database |

There is no offline fallback: without a valid key, seeding and chat fail with a clear error.

## Third-party services and libraries

- **OpenAI API**: Responses API for all agents and the Embeddings API for retrieval and flag confidence. This is the only external service.
- **LangGraph**: agent workflow orchestration.
- **Django + Django REST Framework**: UI pages and JSON API.
- **SQLite**: local storage (embeddings stored as JSON and compared in Python).

## Approach

Each patient message runs through a LangGraph pipeline of small, single-purpose agents rather than one large prompt:

```
intent -> symptom extraction -> longitudinal retrieval -> flag + action update -> responder
```

- **Intent agent** works out what the patient is trying to do this turn (routine check-in, appointment request or change, message to the doctor, or general chat). Patients switch goals mid-conversation constantly (relay a message, chat, then start a check-in), and this was the most reliable way to keep the rest of the workflow and the responder focused on the right thing.
- **Symptom agent** extracts structured observations (status, onset, severity, matching watch-for) and asks follow-ups for missing details one at a time.
- **Retrieval** finds the patient's earlier reports of the same symptom using a hybrid score: of similarity of the symptom text and similarity of the original chat message, with a 0.6 threshold, plus a hard match on the same watch-for so a differently worded "resolved" report isn't missed.
- **Flag agent** checks present or uncertain symptoms against the care team's watch-for guidance and raises alerts to the dashboard. Uses the same hybrid score as what is used for retrival, but there is no threshold, the LLM decides the final action.
- **Action agent** captures appointment and doctor-message requests and any details the patient adds afterward.
- **Responder** calls an LLM to write the reply from a prompt with input from previous nodes.

Stack choices: LangGraph because it's what I'm most familiar with for agent workflows; Django because it matches what I understand the DTxPlus team uses; SQLite for simplicity (models stay PostgreSQL-compatible).

### Assumptions

- One seeded patient (Jordan Ellis, 71, CHF and COPD). Everything patient-specific lives in `Patient.profile_data`, so agent code has no patient-specific logic, and should be compatible with another patient. Evals would likely need to be expanded to account for other watch-fors and their guidance.
- Confidence scores and the 0.6 threshold are informational for now, to be calibrated with clinical/sample data.
- No authentication or provisioning, etc. This is a local prototype only.


## How I used AI tools

I used **GitHub Copilot** and **Claude Code** as coding agents, in a very iterative loop. I designed the architecture myself and used the agents mainly to implement it, debug it, and build out tests.

1. **Design first.** I sketched the initial agents and database schema and wrote them up as a detailed README spec, then had Copilot build the first version from it.
2. **Use it, find edge cases, revise the design.** I spent most of my time chatting with each version as a patient. To see what was happening, I added an agent-flow debug pane to the chat UI (marked `[TEMP DEBUG]`) that shows each node's inputs and outputs and the final responder prompt. That made it clear which nodes were firing correctly and which weren't, so I could make targeted fixes myself or give the coding agent precise instructions. Along the way I kept notes on fields and actions I discovered I needed, and redrew the agent node workflow as a diagram in Google Slides (low tech, I know).
3. **Rewrite the spec and rebuild.** Once the design settled, I wrote a new README defining the final workflow, every agent's actions, and the schema, and used Claude Code to bring the codebase in line with it.
4. **Evals.** I built the eval suite with Claude Code, then added my own cases as I found new edge cases or earlier failures. I also had Claude Code generate a labeled synthetic dataset (`retrieval_eval/`) to test whether my 0.6 retrieval threshold held up. It was mostly sound, so given the time constraints (didn't want to go down too many rabbit holes), I left it as is.

### Example prompts

> Here is my updated README defining the full agent workflow, each agent's inputs, outputs and actions, and the database schema. Update the codebase to match it exactly, including migrations and the seed command, and keep all patient-specific data in `profile_data`.

> I set the retrieval/flag match threshold to 0.6 somewhat arbitrarily. Build a small labeled synthetic dataset of patient symptom histories and run an experiment that measures recall of history rows at that threshold, sweeps the threshold and the 0.7/0.3 weights, and saves charts to `retrieval_eval/results/`.

> In the debug pane, when the patient asks to message the doctor and then starts a routine check-in, the intent node stays on `contact_doctor` and the responder keeps asking about the message. Fix intent handling so each turn is classified on its own meaning, edit the responder system message, and add an eval case for this sequence.

### Notable corrections

- **Intent switching.** Early versions let the workflow get stuck on one intent, or slid into a routine check-in when the patient was just chatting. I added a dedicated intent agent and a code guard so a check-in only starts when the patient explicitly asks for one.
- **Follow-up loops.** The generated code let the model produce its own list of missing symptom details as free text, which could never be cleared, so the assistant kept asking. Missing fields are now computed in code (onset and severity only), and after two follow-ups the observation is saved anyway.
- **Overclaiming in replies.** The responder would say things like "your appointment is booked" or "I've let your doctor know" when nothing had been sent. I added deterministic turn rules to the prompt stating exactly what happened this turn, plus eval checks for those phrases.
- **Silent fallbacks.** Generated code fell back to keyword heuristics when an LLM call failed, which hid real failures. I removed every fallback: a failed call now raises, rolls back the whole turn, and returns the patient's message to the composer to resend.
- **Retrieval misses.** Pure similarity missed later "resolved" reports worded differently from the original complaint ("my legs are back to normal"). I added the watch-for match and history expansion, confirmed on the synthetic benchmark.

## What I'd do next

1. **Validate retrieval and flagging properly.** The 0.7/0.3 weights and 0.6 threshold are mostly educated guesses. I'd build a larger, more realistic labeled dataset (synthetic if no real one is available) and compare retrieval and ranking methods. I'd also look at alternatives to storing embeddings as JSON, such as pgvector, balancing live-chat latency against storage cost.
2. **Tune prompts and models per node.** I didn't optimize prompts. I'd compare prompt styles (including few-shot), models, and providers for each agent, and measure accuracy against latency and cost.
3. **Better responsiveness** Stream responder output to cut latency, as I know this kind of latency would feel unnatural in a voice chat agent.
