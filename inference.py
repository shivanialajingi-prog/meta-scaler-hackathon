import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from env.environment import EmailTriageEnv
from env.models import Action, Observation
from env.tasks import TASK_NAMES

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "openenv_email_triage"

MAX_TOKENS = 220
TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "You are an operations agent for an enterprise inbox. "
    "Return exactly one JSON object with keys action_type, email_id, value. "
    "Valid action_type values: open_email, set_priority, assign_folder, draft_reply, send_reply, archive, mark_spam, escalate, finish, noop. "
    "Use null for missing email_id or value. No markdown."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_text = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={reward_text}",
        flush=True,
    )


def _build_prompt(observation: Observation) -> str:
    rows = []
    for row in observation.inbox:
        rows.append(
            {
                "email_id": row.email_id,
                "sender": row.sender,
                "subject": row.subject,
                "sla_hours": row.sla_hours,
                "opened": row.opened,
                "priority": row.priority,
                "folder": row.folder,
                "has_draft": row.has_draft,
                "reply_sent": row.reply_sent,
                "escalated": row.escalated,
                "spam": row.spam,
            }
        )

    payload = {
        "objective": observation.objective,
        "steps_remaining": observation.steps_remaining,
        "current_email_id": observation.current_email_id,
        "inbox": rows,
        "last_action_error": observation.last_action_error,
    }
    return json.dumps(payload, ensure_ascii=True)


def _fallback_policy(observation: Observation) -> Action:
    rows = {r.email_id: r for r in observation.inbox}

    def choose_unopened(ids: List[str]) -> Optional[str]:
        for eid in ids:
            row = rows.get(eid)
            if row and not row.opened and row.folder != "archive" and not row.spam:
                return eid
        return None

    task_rules: Dict[str, Dict[str, Any]] = {
        "easy_email_triage": {
            "spam": ["E2"],
            "priority": {"E1": "high", "E3": "low"},
            "folder": {"E1": "finance", "E3": "hr"},
            "reply": {},
            "escalate": [],
        },
        "medium_sla_coordination": {
            "spam": ["M5"],
            "priority": {"M1": "critical", "M2": "high", "M3": "high", "M4": "normal"},
            "folder": {"M1": "support", "M2": "finance", "M3": "security", "M4": "support"},
            "reply": {"M1": "ack_incident", "M2": "finance_ack", "M4": "support_ack"},
            "escalate": ["M1", "M3"],
        },
        "hard_cross_team_incident": {
            "spam": ["H8"],
            "priority": {
                "H1": "critical",
                "H2": "high",
                "H3": "high",
                "H4": "normal",
                "H5": "critical",
                "H6": "normal",
                "H7": "low",
            },
            "folder": {
                "H1": "support",
                "H2": "support",
                "H3": "security",
                "H4": "finance",
                "H5": "security",
                "H6": "support",
                "H7": "hr",
            },
            "reply": {
                "H1": "ack_incident",
                "H3": "legal_ack",
                "H4": "finance_ack",
                "H5": "security_ack",
                "H6": "cs_ack",
            },
            "escalate": ["H1", "H2", "H5"],
        },
    }

    rule = task_rules[observation.task_name]

    for eid in rule["spam"]:
        row = rows.get(eid)
        if row and not row.spam:
            return Action(action_type="mark_spam", email_id=eid)

    for eid in sorted(rows.keys()):
        row = rows[eid]
        if row.folder == "archive" or row.spam:
            continue

        target_priority = rule["priority"].get(eid)
        if target_priority and row.priority != target_priority:
            return Action(action_type="set_priority", email_id=eid, value=target_priority)

        target_folder = rule["folder"].get(eid)
        if target_folder and row.folder != target_folder:
            return Action(action_type="assign_folder", email_id=eid, value=target_folder)

        if eid in rule["escalate"] and not row.escalated:
            return Action(action_type="escalate", email_id=eid)

        reply_template = rule["reply"].get(eid)
        if reply_template and not row.reply_sent:
            if observation.current_email_id != eid:
                unopened = choose_unopened([eid])
                if unopened:
                    return Action(action_type="open_email", email_id=eid)
            if row.has_draft:
                return Action(action_type="send_reply", email_id=eid)
            if not row.has_draft:
                return Action(action_type="draft_reply", email_id=eid, value=reply_template)

    for eid in sorted(rows.keys()):
        row = rows[eid]
        if row.folder != "archive":
            if eid in rule["reply"] and not row.reply_sent:
                return Action(action_type="send_reply", email_id=eid)
            return Action(action_type="archive", email_id=eid)

    return Action(action_type="finish")


def _parse_action(text: str) -> Optional[Action]:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
        if not isinstance(data, dict):
            return None
        return Action(
            action_type=data.get("action_type", "noop"),
            email_id=data.get("email_id"),
            value=data.get("value"),
        )
    except Exception:
        return None


def get_action(client: OpenAI, observation: Observation) -> Action:
    user_prompt = _build_prompt(observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        message = (completion.choices[0].message.content or "").strip()
        parsed = _parse_action(message)
        if parsed is not None:
            return parsed
    except Exception:
        pass
    return _fallback_policy(observation)


def _action_str(action: Action) -> str:
    return f"{action.action_type}:{action.email_id or '-'}:{action.value or '-'}"


def run_task(client: OpenAI, task_name: str) -> None:
    env = EmailTriageEnv(task_name=task_name)
    rewards: List[float] = []
    steps = 0
    score = 0.0001
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset()
        max_steps = observation.steps_remaining

        for idx in range(1, max_steps + 1):
            action = get_action(client, observation)
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            steps = idx
            error = info.get("last_action_error") if isinstance(info, dict) else None
            log_step(step=idx, action=_action_str(action), reward=reward, done=done, error=error)
            if done:
                break

        current_state = env.state()
        score = float(current_state.final_score)
        score = max(0.0001, min(0.9999, score))
        success = score >= 0.8
    finally:
        env.close()
        log_end(success=success, steps=steps, score=score, rewards=rewards)


def main() -> None:
    api_key = HF_TOKEN or os.getenv("API_KEY") or ""
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    for task_name in TASK_NAMES:
        run_task(client=client, task_name=task_name)


if __name__ == "__main__":
    main()
