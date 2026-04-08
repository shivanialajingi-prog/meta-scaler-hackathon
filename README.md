---
title: Meta Scalar Hackathon
emoji: 📬
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
---

# OpenEnv Email Triage Environment

## Problem Description
This project provides a production-style OpenEnv environment that simulates enterprise inbox operations under SLA and cross-team constraints. The agent must process realistic email traffic, prioritize urgent work, route issues to the correct function, send required acknowledgments, escalate critical incidents, and close work cleanly.

## Real-World Motivation
Email triage is a high-frequency workflow in support, security, and operations teams. Poor prioritization creates SLA breaches and customer churn. This environment captures practical trade-offs between speed, correctness, and efficiency so RL and LLM agents can be evaluated in a deterministic, reproducible setting.

## Environment Interface
Implemented in `env/environment.py` as `EmailTriageEnv`:
- `reset() -> Observation`
- `step(action: Action) -> (Observation, reward, done, info)`
- `state() -> State`

Pydantic models are defined in `env/models.py`:
- `Action`
- `Observation`
- `State`
- `RewardBreakdown`

## Action Space
`Action.action_type` supports:
- `open_email`
- `set_priority`
- `assign_folder`
- `draft_reply`
- `send_reply`
- `archive`
- `mark_spam`
- `escalate`
- `finish`
- `noop`

Action payload fields:
- `email_id` (optional; required for most operations)
- `value` (optional; used for priority/folder/template values)

## Observation Space
`Observation` contains:
- `task_name`
- `objective`
- `steps_remaining`
- `current_email_id`
- `inbox` (structured rows with SLA, priority, status, escalation and reply flags)
- `current_email` (full selected email payload)
- `last_action_error`

## Tasks
Three deterministic tasks with increasing difficulty are implemented in `env/tasks.py`.

1. `easy_email_triage`
- Short horizon triage with one urgent finance email, one spam item, one low-priority internal email.
- Focus: basic routing, spam handling, and closure.

2. `medium_sla_coordination`
- Mixed support/security/finance inbox with short SLA windows.
- Focus: multi-step sequencing with acknowledgments and selective escalation.

3. `hard_cross_team_incident`
- Long-horizon incident day with concurrent infra, legal, security, finance, and customer workflows.
- Focus: planning under constraints, multi-objective execution, and operational efficiency.

## Grading and Determinism
Deterministic graders are in `env/graders.py`.
- Score range is always `0.0` to `1.0`.
- No randomness is used.
- Weighted metrics: priority, folder routing, spam handling, closure, required replies, and escalations.
- Efficiency penalties apply for invalid/loop/no-op behavior.

## Reward Logic
Dense rewards are implemented in `env/reward.py`.
- Progress-based shaping: reward tracks incremental completion improvements each step.
- Action-quality reward for productive actions.
- Penalties for:
  - invalid actions
  - repeated loops
  - no-ops and wasted steps
- Additional urgency penalty in medium/hard tasks when urgent items remain unopened.
- Terminal bonus tied to final deterministic task score.

## Project Structure
```text
project/
├── openenv.yaml
├── env/
│   ├── __init__.py
│   ├── api.py
│   ├── environment.py
│   ├── models.py
│   ├── tasks.py
│   ├── graders.py
│   └── reward.py
├── inference.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup Instructions
```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Run Environment API (HF Space Compatible)
```bash
uvicorn env.api:app --host 0.0.0.0 --port 7860
```

Endpoints:
- `POST /reset` with JSON: `{"task_name":"easy_email_triage"}`
- `POST /step` with JSON: `{"action":{"action_type":"open_email","email_id":"E1","value":null}}`
- `GET /state`

## Run Inference
The script is in project root and uses the OpenAI client with required environment variables.

Required env vars:
- `HF_TOKEN`
- `API_BASE_URL` (default provided)
- `MODEL_NAME` (default provided)

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

`inference.py` emits only structured logs in this format:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

and executes all three tasks in sequence.

## Baseline Scores
Deterministic fallback policy baseline (no external model dependency):
- `easy_email_triage`: `1.00`
- `medium_sla_coordination`: `1.00`
- `hard_cross_team_incident`: `1.00`

These values are reproducible for the included fallback policy and deterministic task logic.
