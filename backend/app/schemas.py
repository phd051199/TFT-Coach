from __future__ import annotations

from pydantic import BaseModel, Field


class CoachRequest(BaseModel):
    level: int = Field(default=4, ge=3, le=10)
    ownedChampionIds: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)

