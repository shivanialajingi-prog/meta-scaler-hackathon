from __future__ import annotations

from typing import Dict, List

from .models import EmailState, TaskSpec

TASK_NAMES = [
    "easy_email_triage",
    "medium_sla_coordination",
    "hard_cross_team_incident",
]


def _easy_task() -> TaskSpec:
    emails: List[EmailState] = [
        EmailState(
            email_id="E1",
            sender="billing@vendor.com",
            subject="Invoice 48321 overdue notice",
            body="Payment is due today. Please confirm payment date.",
            received_hour=9,
            sla_hours=8,
        ),
        EmailState(
            email_id="E2",
            sender="promo@cheap-travel.biz",
            subject="You won 2 free flights",
            body="Click this link to claim your reward now.",
            received_hour=10,
            sla_hours=None,
        ),
        EmailState(
            email_id="E3",
            sender="teammate@company.com",
            subject="Lunch plans",
            body="Are we still on for lunch tomorrow?",
            received_hour=11,
            sla_hours=48,
        ),
    ]
    expectations: Dict[str, object] = {
        "required_priorities": {"E1": "high", "E3": "low"},
        "required_folders": {"E1": "finance", "E3": "hr"},
        "required_spam": ["E2"],
        "required_archived": ["E1", "E2", "E3"],
        "required_replies": {},
        "required_escalations": [],
    }
    return TaskSpec(
        name="easy_email_triage",
        difficulty="easy",
        objective=(
            "Triage three inbox items. Mark obvious spam, route business mail to correct folders, "
            "set practical priorities, and archive all handled items."
        ),
        max_steps=10,
        emails=emails,
        expectations=expectations,
    )


def _medium_task() -> TaskSpec:
    emails: List[EmailState] = [
        EmailState(
            email_id="M1",
            sender="enterprise-customer@acme.io",
            subject="SLA breach warning for ticket #7731",
            body="Our production issue is still unresolved. We need ETA within 1 hour.",
            received_hour=8,
            sla_hours=1,
        ),
        EmailState(
            email_id="M2",
            sender="finance-lead@company.com",
            subject="Quarter close approval needed",
            body="Need approval for journal adjustment before noon.",
            received_hour=9,
            sla_hours=3,
        ),
        EmailState(
            email_id="M3",
            sender="security-bot@alerts.io",
            subject="Suspicious login from new geo",
            body="User account login from unknown country requires review.",
            received_hour=9,
            sla_hours=2,
        ),
        EmailState(
            email_id="M4",
            sender="support@smallclient.net",
            subject="Password reset not arriving",
            body="Customer cannot receive password reset email.",
            received_hour=10,
            sla_hours=6,
        ),
        EmailState(
            email_id="M5",
            sender="newsletter@events.org",
            subject="Weekly webinar lineup",
            body="This week events and sponsor promotions.",
            received_hour=11,
            sla_hours=None,
        ),
    ]
    expectations: Dict[str, object] = {
        "required_priorities": {"M1": "critical", "M2": "high", "M3": "high", "M4": "normal"},
        "required_folders": {"M1": "support", "M2": "finance", "M3": "security", "M4": "support"},
        "required_spam": ["M5"],
        "required_archived": ["M1", "M2", "M3", "M4", "M5"],
        "required_replies": {
            "M1": "ack_incident",
            "M2": "finance_ack",
            "M4": "support_ack",
        },
        "required_escalations": ["M1", "M3"],
    }
    return TaskSpec(
        name="medium_sla_coordination",
        difficulty="medium",
        objective=(
            "Coordinate mixed-priority inbox under SLA pressure: route to teams, send acknowledgments, "
            "escalate critical risk, and close all items with minimal wasted actions."
        ),
        max_steps=28,
        emails=emails,
        expectations=expectations,
    )


def _hard_task() -> TaskSpec:
    emails: List[EmailState] = [
        EmailState(
            email_id="H1",
            sender="cio@enterprise-a.com",
            subject="P1 outage escalation: payment API down",
            body="Global payment failures detected. Need immediate response bridge.",
            received_hour=7,
            sla_hours=1,
        ),
        EmailState(
            email_id="H2",
            sender="oncall@infra.company.com",
            subject="Database replica lag > 10m",
            body="Replica lag could impact reporting and failover readiness.",
            received_hour=7,
            sla_hours=2,
        ),
        EmailState(
            email_id="H3",
            sender="legal@company.com",
            subject="Urgent data deletion request (GDPR)",
            body="Customer requests deletion confirmation within 24 hours.",
            received_hour=8,
            sla_hours=4,
        ),
        EmailState(
            email_id="H4",
            sender="payroll@company.com",
            subject="Payroll export mismatch",
            body="Potential mismatch in payroll tax totals.",
            received_hour=8,
            sla_hours=5,
        ),
        EmailState(
            email_id="H5",
            sender="security@vendor-alerts.com",
            subject="Critical CVE in production dependency",
            body="CVE score 9.8 affecting deployed package.",
            received_hour=9,
            sla_hours=3,
        ),
        EmailState(
            email_id="H6",
            sender="customer-success@company.com",
            subject="Top account renewal at risk",
            body="Account requires executive check-in today.",
            received_hour=10,
            sla_hours=6,
        ),
        EmailState(
            email_id="H7",
            sender="benefits@company.com",
            subject="Open enrollment reminder",
            body="Reminder for internal team only.",
            received_hour=11,
            sla_hours=48,
        ),
        EmailState(
            email_id="H8",
            sender="news@random-blog.net",
            subject="Top 10 crypto picks",
            body="Promotional content with affiliate links.",
            received_hour=12,
            sla_hours=None,
        ),
    ]
    expectations: Dict[str, object] = {
        "required_priorities": {
            "H1": "critical",
            "H2": "high",
            "H3": "high",
            "H4": "normal",
            "H5": "critical",
            "H6": "normal",
            "H7": "low",
        },
        "required_folders": {
            "H1": "support",
            "H2": "support",
            "H3": "security",
            "H4": "finance",
            "H5": "security",
            "H6": "support",
            "H7": "hr",
        },
        "required_spam": ["H8"],
        "required_archived": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"],
        "required_replies": {
            "H1": "ack_incident",
            "H3": "legal_ack",
            "H4": "finance_ack",
            "H5": "security_ack",
            "H6": "cs_ack",
        },
        "required_escalations": ["H1", "H2", "H5"],
    }
    return TaskSpec(
        name="hard_cross_team_incident",
        difficulty="hard",
        objective=(
            "Run cross-functional inbox operations for an active incident day. Handle compliance, infra, security, "
            "finance, and customer pressure while preserving SLA priority and avoiding redundant actions."
        ),
        max_steps=42,
        emails=emails,
        expectations=expectations,
    )


def get_task(task_name: str) -> TaskSpec:
    task_map = {
        "easy_email_triage": _easy_task,
        "medium_sla_coordination": _medium_task,
        "hard_cross_team_incident": _hard_task,
    }
    if task_name not in task_map:
        raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {', '.join(TASK_NAMES)}")
    return task_map[task_name]()
