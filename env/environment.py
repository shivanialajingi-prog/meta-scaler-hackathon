from __future__ import annotations

from typing import Dict, Optional, Tuple

from .graders import grade_task
from .models import Action, EmailState, InboxRow, Observation, State, TaskSpec
from .reward import compute_step_reward
from .tasks import TASK_NAMES, get_task

REPLY_TEMPLATES = {
    "ack_incident": "Acknowledged. Incident bridge initiated. Next update in 30 minutes.",
    "finance_ack": "Acknowledged. Finance review queued and owner assigned.",
    "support_ack": "Acknowledged. Support troubleshooting started and ETA will follow.",
    "security_ack": "Acknowledged. Security triage started; containment in progress.",
    "legal_ack": "Acknowledged. Legal workflow initiated and deletion verification scheduled.",
    "cs_ack": "Acknowledged. Account team preparing executive outreach today.",
}


class EmailTriageEnv:
    """Deterministic OpenEnv-compatible email triage environment."""

    def __init__(self, task_name: str = "easy_email_triage") -> None:
        if task_name not in TASK_NAMES:
            raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {', '.join(TASK_NAMES)}")
        self.task_name = task_name
        self.task: TaskSpec = get_task(task_name)
        self._state: Optional[State] = None

    def set_task(self, task_name: str) -> None:
        if task_name not in TASK_NAMES:
            raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {', '.join(TASK_NAMES)}")
        self.task_name = task_name
        self.task = get_task(task_name)
        self._state = None

    def reset(self) -> Observation:
        emails = {email.email_id: email.model_copy(deep=True) for email in self.task.emails}
        self._state = State(
            task_name=self.task.name,
            objective=self.task.objective,
            step_count=0,
            max_steps=self.task.max_steps,
            done=False,
            current_email_id=None,
            emails=emails,
            action_history=[],
            invalid_actions=0,
            loop_actions=0,
            noop_actions=0,
            wasted_steps=0,
            cumulative_reward=0.0,
            final_score=0.0,
            last_action_error=None,
        )
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, object]]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        if self._state.done:
            obs = self._build_observation()
            return obs, 0.0, True, {
                "last_action_error": "episode_already_done",
                "score": self._state.final_score,
                "reward_breakdown": {"total": 0.0, "progress": 0.0, "quality": 0.0, "penalty": 0.0},
            }

        previous_state = self._state.model_copy(deep=True)
        action_text = self._action_to_string(action)
        looped = len(self._state.action_history) >= 2 and self._state.action_history[-1] == action_text == self._state.action_history[-2]

        action_valid, error_message = self._apply_action(action)
        self._state.step_count += 1
        self._state.action_history.append(action_text)
        self._state.last_action_error = error_message

        if not action_valid:
            self._state.invalid_actions += 1
            self._state.wasted_steps += 1

        if looped:
            self._state.loop_actions += 1

        no_op = self._state.emails == previous_state.emails and self._state.current_email_id == previous_state.current_email_id and action.action_type != "finish"
        if no_op:
            self._state.noop_actions += 1
            self._state.wasted_steps += 1

        if action.action_type == "finish":
            self._state.done = True
        if self._state.step_count >= self._state.max_steps:
            self._state.done = True

        if self._state.done:
            self._state.final_score = grade_task(self.task, self._state)

        reward_parts = compute_step_reward(
            task=self.task,
            previous_state=previous_state,
            current_state=self._state,
            action=action,
            action_valid=action_valid,
            looped=looped,
            no_op=no_op,
        )
        self._state.cumulative_reward += reward_parts.total

        obs = self._build_observation()
        info: Dict[str, object] = {
            "last_action_error": self._state.last_action_error,
            "score": self._state.final_score if self._state.done else grade_task(self.task, self._state),
            "reward_breakdown": reward_parts.model_dump(),
            "step_count": self._state.step_count,
            "max_steps": self._state.max_steps,
        }
        return obs, reward_parts.total, self._state.done, info

    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() before state().")
        return self._state.model_copy(deep=True)

    def close(self) -> None:
        return None

    def _build_observation(self) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() before observation build.")

        inbox = [
            InboxRow(
                email_id=e.email_id,
                sender=e.sender,
                subject=e.subject,
                sla_hours=e.sla_hours,
                opened=e.opened,
                priority=e.priority,
                folder=e.folder,
                status=e.status,
                has_draft=e.draft_reply is not None,
                reply_sent=e.reply_sent,
                escalated=e.escalated,
                spam=e.spam,
            )
            for e in self._state.emails.values()
        ]
        current_email = self._state.emails.get(self._state.current_email_id) if self._state.current_email_id else None

        return Observation(
            task_name=self._state.task_name,
            objective=self._state.objective,
            steps_remaining=max(0, self._state.max_steps - self._state.step_count),
            current_email_id=self._state.current_email_id,
            inbox=inbox,
            current_email=current_email.model_copy(deep=True) if current_email else None,
            last_action_error=self._state.last_action_error,
        )

    def _apply_action(self, action: Action) -> Tuple[bool, Optional[str]]:
        assert self._state is not None

        if action.action_type == "noop":
            return True, None

        target: Optional[EmailState] = None
        if action.email_id:
            target = self._state.emails.get(action.email_id)
            if target is None:
                return False, "unknown_email_id"

        if action.action_type == "open_email":
            if target is None:
                return False, "open_email_requires_email_id"
            if target.status == "closed":
                return False, "cannot_open_closed_email"
            target.opened = True
            target.status = "active"
            self._state.current_email_id = target.email_id
            return True, None

        if action.action_type == "set_priority":
            if target is None or action.value is None:
                return False, "set_priority_requires_email_id_and_value"
            if action.value not in {"low", "normal", "high", "critical"}:
                return False, "invalid_priority"
            target.priority = action.value  # type: ignore[assignment]
            target.status = "active"
            return True, None

        if action.action_type == "assign_folder":
            if target is None or action.value is None:
                return False, "assign_folder_requires_email_id_and_value"
            if action.value not in {"inbox", "finance", "support", "security", "hr", "spam", "archive"}:
                return False, "invalid_folder"
            target.folder = action.value  # type: ignore[assignment]
            target.routed_folder = action.value  # type: ignore[assignment]
            target.status = "active"
            return True, None

        if action.action_type == "draft_reply":
            if target is None or action.value is None:
                return False, "draft_reply_requires_email_id_and_template"
            if target.spam:
                return False, "cannot_reply_to_spam"
            if action.value not in REPLY_TEMPLATES:
                return False, "unknown_reply_template"
            target.draft_reply = action.value
            target.status = "active"
            return True, None

        if action.action_type == "send_reply":
            if target is None:
                return False, "send_reply_requires_email_id"
            if target.spam:
                return False, "cannot_reply_to_spam"
            if not target.draft_reply:
                return False, "draft_required_before_send"
            target.reply_sent = True
            target.status = "active"
            return True, None

        if action.action_type == "archive":
            if target is None:
                return False, "archive_requires_email_id"
            target.folder = "archive"
            target.status = "closed"
            if self._state.current_email_id == target.email_id:
                self._state.current_email_id = None
            return True, None

        if action.action_type == "mark_spam":
            if target is None:
                return False, "mark_spam_requires_email_id"
            target.spam = True
            target.folder = "spam"
            target.routed_folder = "spam"
            target.status = "closed"
            target.reply_sent = False
            if self._state.current_email_id == target.email_id:
                self._state.current_email_id = None
            return True, None

        if action.action_type == "escalate":
            if target is None:
                return False, "escalate_requires_email_id"
            if target.spam:
                return False, "cannot_escalate_spam"
            target.escalated = True
            target.status = "active"
            return True, None

        if action.action_type == "finish":
            return True, None

        return False, "unknown_action_type"

    @staticmethod
    def _action_to_string(action: Action) -> str:
        return f"{action.action_type}:{action.email_id or '-'}:{action.value or '-'}"
