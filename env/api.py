from __future__ import annotations

from fastapi import Body, FastAPI
from pydantic import BaseModel

from .environment import EmailTriageEnv
from .models import Action
from .tasks import TASK_NAMES

app = FastAPI(title="OpenEnv Email Triage", version="0.1.0")
_env = EmailTriageEnv(task_name="easy_email_triage")


class ResetRequest(BaseModel):
    task_name: str = "easy_email_triage"


class StepRequest(BaseModel):
    action: Action


@app.get("/")
def root() -> dict:
    return {
        "name": "OpenEnv Email Triage API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "tasks": TASK_NAMES}


@app.post("/reset")
def reset(payload: ResetRequest | None = Body(default=None)) -> dict:
    task_name = payload.task_name if payload else "easy_email_triage"
    _env.set_task(task_name)
    obs = _env.reset()
    return {"observation": obs.model_dump(), "done": False}


@app.post("/step")
def step(payload: StepRequest) -> dict:
    obs, reward, done, info = _env.step(payload.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict:
    return _env.state().model_dump()
