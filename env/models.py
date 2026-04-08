from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ActionType = Literal[
    "open_email",
    "set_priority",
    "assign_folder",
    "draft_reply",
    "send_reply",
    "archive",
    "mark_spam",
    "escalate",
    "finish",
    "noop",
]

Priority = Literal["low", "normal", "high", "critical"]
Folder = Literal["inbox", "finance", "support", "security", "hr", "spam", "archive"]
Status = Literal["new", "active", "closed"]


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    email_id: Optional[str] = None
    value: Optional[str] = None


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: float
    progress: float
    quality: float
    penalty: float


class EmailState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email_id: str
    sender: str
    subject: str
    body: str
    received_hour: int = Field(ge=0, le=23)
    sla_hours: Optional[int] = Field(default=None, ge=1)
    opened: bool = False
    priority: Priority = "normal"
    folder: Folder = "inbox"
    routed_folder: Folder = "inbox"
    status: Status = "new"
    draft_reply: Optional[str] = None
    reply_sent: bool = False
    escalated: bool = False
    spam: bool = False


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    max_steps: int = Field(ge=3)
    emails: List[EmailState]
    expectations: Dict[str, object]


class InboxRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email_id: str
    sender: str
    subject: str
    sla_hours: Optional[int]
    opened: bool
    priority: Priority
    folder: Folder
    status: Status
    has_draft: bool
    reply_sent: bool
    escalated: bool
    spam: bool


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str
    objective: str
    steps_remaining: int
    current_email_id: Optional[str]
    inbox: List[InboxRow]
    current_email: Optional[EmailState]
    last_action_error: Optional[str] = None


class State(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str
    objective: str
    step_count: int
    max_steps: int
    done: bool
    current_email_id: Optional[str]
    emails: Dict[str, EmailState]
    action_history: List[str]
    invalid_actions: int
    loop_actions: int
    noop_actions: int
    wasted_steps: int
    cumulative_reward: float
    final_score: float = Field(default=0.0001, gt=0.0, lt=1.0)
    last_action_error: Optional[str] = None
