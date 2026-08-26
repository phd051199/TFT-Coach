from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class TrainingSample:
    source: str
    patch: str
    region: str
    units: list[str]
    items: list[str] = field(default_factory=list)
    item_holders: dict[str, list[str]] = field(default_factory=dict)
    traits: list[str] = field(default_factory=list)
    level: int = 8
    avg_placement: float = 4.5
    top4_rate: float | None = None
    win_rate: float | None = None
    games: int = 1
    source_weight: float = 1.0
    sample_kind: str = "final_board"
    context_id: str | None = None

    @property
    def target_strength(self) -> float:
        # Placement is always available. Optional aggregate rates refine the label without
        # letting any site's definition of a "tier" become the ground truth.
        placement_strength = max(0.0, min(1.0, (8.5 - self.avg_placement) / 7.5))
        values = [(placement_strength, 0.72)]
        if self.top4_rate is not None:
            values.append((max(0.0, min(1.0, self.top4_rate)), 0.20))
        if self.win_rate is not None:
            values.append((max(0.0, min(1.0, self.win_rate)), 0.08))
        denominator = sum(weight for _, weight in values)
        return sum(value * weight for value, weight in values) / denominator

    @property
    def training_weight(self) -> float:
        # Aggregate rows with more games are more trustworthy, but cap the influence so a
        # giant site cannot drown out every other region/source.
        evidence = 0.45 + min(1.55, math.log1p(max(1, self.games)) / 5.0)
        return max(0.01, self.source_weight * evidence)

    def to_json(self) -> dict:
        value = asdict(self)
        value["target_strength"] = self.target_strength
        value["training_weight"] = self.training_weight
        return value


class DataSource(Protocol):
    name: str

    async def collect(self) -> list[TrainingSample]: ...


def write_jsonl(path: Path, rows: list[TrainingSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
