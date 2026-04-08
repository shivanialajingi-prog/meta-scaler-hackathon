from __future__ import annotations

from .graders import compute_completion_breakdown, grade_task
from .models import Action, RewardBreakdown, State, TaskSpec


def _progress_scalar(task: TaskSpec, state: State) -> float:
    metrics = compute_completion_breakdown(task, state)
    return (
        (metrics["priority"] * 0.20)
        + (metrics["folder"] * 0.20)
        + (metrics["spam"] * 0.10)
        + (metrics["archive"] * 0.20)
        + (metrics["reply"] * 0.20)
        + (metrics["escalation"] * 0.10)
    )


def compute_step_reward(
    task: TaskSpec,
    previous_state: State,
    current_state: State,
    action: Action,
    action_valid: bool,
    looped: bool,
    no_op: bool,
) -> RewardBreakdown:
    prev_progress = _progress_scalar(task, previous_state)
    curr_progress = _progress_scalar(task, current_state)

    progress_component = (curr_progress - prev_progress) * 1.6

    quality_component = 0.0
    if action.action_type in {"set_priority", "assign_folder", "draft_reply", "send_reply", "escalate", "archive", "mark_spam"} and action_valid:
        quality_component += 0.03
    if action.action_type == "finish" and current_state.done:
        quality_component += 0.02

    penalty_component = -0.01
    if not action_valid:
        penalty_component -= 0.20
    if looped:
        penalty_component -= 0.08
    if no_op:
        penalty_component -= 0.06

    # Time pressure constraints in harder tasks: leaving low-SLA items untouched should hurt gradually.
    if task.difficulty in {"medium", "hard"}:
        urgent_unopened = sum(
            1
            for email in current_state.emails.values()
            if email.sla_hours is not None and email.sla_hours <= 2 and not email.opened
        )
        penalty_component -= min(0.06, urgent_unopened * 0.02)

    terminal_bonus = 0.0
    if current_state.done:
        final_score = grade_task(task, current_state)
        terminal_bonus = 0.40 * final_score

    total = progress_component + quality_component + penalty_component + terminal_bonus
    total = max(-1.0, min(1.0, total))

    return RewardBreakdown(
        total=round(total, 4),
        progress=round(progress_component, 4),
        quality=round(quality_component + terminal_bonus, 4),
        penalty=round(penalty_component, 4),
    )
