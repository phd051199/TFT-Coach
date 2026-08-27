from __future__ import annotations

from pydantic import BaseModel, Field


class PreviousItemRecommendation(BaseModel):
    stage: str
    itemId: str
    holderId: str | None = None


class CoachRequest(BaseModel):
    level: int = Field(default=4, ge=3, le=10)
    ownedChampionIds: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    targetCompId: str | None = None
    previousLevel: int | None = Field(default=None, ge=3, le=10)
    previousCompId: str | None = None
    previousOwnedChampionIds: list[str] = Field(default_factory=list)
    previousComponents: list[str] = Field(default_factory=list)
    previousItemPlan: list[PreviousItemRecommendation] = Field(default_factory=list)

