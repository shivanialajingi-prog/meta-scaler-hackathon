from __future__ import annotations

from typing import Dict, Iterable

from .models import State, TaskSpec

_SCORE_EPSILON = 0.0001


def _fraction_match(required: Dict[str, str], observed: Dict[str, str]) -> float:
    if not required:
        return 1.0
    hits = sum(1 for key, expected in required.items() if observed.get(key) == expected)
    return hits / len(required)


def _fraction_contains(required_ids: Iterable[str], observed_ids: Iterable[str]) -> float:
    required = list(required_ids)
    if not required:
        return 1.0
    observed = set(observed_ids)
    hits = sum(1 for item in required if item in observed)
    return hits / len(required)


def compute_completion_breakdown(task: TaskSpec, state: State) -> Dict[str, float]:
    expected_priorities: Dict[str, str] = task.expectations.get("required_priorities", {})  # type: ignore[assignment]
    expected_folders: Dict[str, str] = task.expectations.get("required_folders", {})  # type: ignore[assignment]
    expected_spam = task.expectations.get("required_spam", [])  # type: ignore[assignment]
    expected_archived = task.expectations.get("required_archived", [])  # type: ignore[assignment]
    expected_replies: Dict[str, str] = task.expectations.get("required_replies", {})  # type: ignore[assignment]
    expected_escalations = task.expectations.get("required_escalations", [])  # type: ignore[assignment]

    observed_priorities = {eid: email.priority for eid, email in state.emails.items()}
    observed_folders = {eid: email.routed_folder for eid, email in state.emails.items()}
    observed_spam = [eid for eid, email in state.emails.items() if email.spam]
    observed_archived = [eid for eid, email in state.emails.items() if email.folder == "archive" or email.status == "closed"]
    observed_replies = {eid: (email.draft_reply or "") for eid, email in state.emails.items() if email.reply_sent}
    observed_escalations = [eid for eid, email in state.emails.items() if email.escalated]

    reply_hits = 0
    if expected_replies:
        for email_id, template in expected_replies.items():
            if observed_replies.get(email_id) == template:
                reply_hits += 1
        reply_score = reply_hits / len(expected_replies)
    else:
        reply_score = 1.0

    return {
        "priority": _fraction_match(expected_priorities, observed_priorities),
        "folder": _fraction_match(expected_folders, observed_folders),
        "spam": _fraction_contains(expected_spam, observed_spam),
        "archive": _fraction_contains(expected_archived, observed_archived),
        "reply": reply_score,
        "escalation": _fraction_contains(expected_escalations, observed_escalations),
    }


def grade_task(task: TaskSpec, state: State) -> float:
    metrics = compute_completion_breakdown(task, state)

    weighted_score = (
        (metrics["priority"] * 0.20)
        + (metrics["folder"] * 0.20)
        + (metrics["spam"] * 0.10)
        + (metrics["archive"] * 0.20)
        + (metrics["reply"] * 0.20)
        + (metrics["escalation"] * 0.10)
    )

    efficiency_penalty = min(0.20, (state.invalid_actions * 0.03) + (state.loop_actions * 0.02) + (state.noop_actions * 0.01))
    raw_score = weighted_score - efficiency_penalty
    # Keep task scores strictly inside (0, 1) to satisfy evaluator constraints.
    score = max(_SCORE_EPSILON, min(1.0 - _SCORE_EPSILON, raw_score))
    return float(round(score, 4))
