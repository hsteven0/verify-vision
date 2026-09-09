from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

UpdateState = Literal["not_checked", "checking", "up_to_date", "updates_available", "unavailable"]
UpdateRisk = Literal["critical_runtime", "application"]
UpdateEcosystem = Literal["python", "frontend"]


class DependencyUpdate(BaseModel):
    package: str
    ecosystem: UpdateEcosystem
    risk: UpdateRisk
    current_version: str
    latest_version: str
    release_url: str


class DependencyUpdateStatus(BaseModel):
    state: UpdateState = "not_checked"
    last_checked_at: datetime | None = None
    next_check_at: datetime | None = None
    packages_checked: int = Field(default=0, ge=0)
    updates: list[DependencyUpdate] = Field(default_factory=list)
    detail: str | None = None
