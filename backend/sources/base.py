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
    age_hours: float | None = None
    consensus_sources: int = 1
    agreement: float = 1.0

    @staticmethod
    def _shrink(value: float, games: int, prior: float, prior_games: float) -> float:
        n = max(1.0, float(games))
        return (value * n + prior * prior_games) / (n + prior_games)

    @property
    def target_strength(self) -> float:
        placement_raw = max(0.0, min(1.0, (8.5 - self.avg_placement) / 7.5))
        if self.sample_kind in {
            "final_board",
            "pro_final_board",
            "pro_item_holder_build",
            "high_elo_final_board",
            "high_elo_item_holder_build",
        } and self.games <= 1:
            # A match placement is a noisy but valid per-game outcome. Shrinking its label to
            # neutral destroys ranking information; reliability belongs in sample_weight.
            return placement_raw
        # Shrink noisy/small samples toward neutral TFT priors. This prevents a 3-game
        # 1.0-average board from becoming a stronger label than a board seen thousands of
        # times across live sources.
        avg = self._shrink(self.avg_placement, self.games, 4.5, 28.0)
        placement_strength = max(0.0, min(1.0, (8.5 - avg) / 7.5))
        values = [(placement_strength, 0.72)]
        if self.top4_rate is not None:
            top4 = self._shrink(max(0.0, min(1.0, self.top4_rate)), self.games, 0.5, 36.0)
            values.append((top4, 0.20))
        if self.win_rate is not None:
            win = self._shrink(max(0.0, min(1.0, self.win_rate)), self.games, 0.125, 48.0)
            values.append((win, 0.08))
        denominator = sum(weight for _, weight in values)
        return sum(value * weight for value, weight in values) / denominator

    @property
    def training_weight(self) -> float:
        # Aggregate rows with more games are more trustworthy, but cap the influence so a
        # giant site cannot drown out every other region/source.
        evidence = 0.45 + min(1.55, math.log1p(max(1, self.games)) / 5.0)
        cross_source = min(1.30, 1.0 + max(0, self.consensus_sources - 1) * 0.12)
        agreement = max(0.55, min(1.0, self.agreement))
        freshness = 1.0
        if self.age_hours is not None:
            # Meta moves quickly after a patch. Do not delete older rows entirely, but make
            # fresh live observations materially more important.
            freshness = max(0.55, math.exp(-max(0.0, self.age_hours) / (24.0 * 6.0)))
        return max(0.01, self.source_weight * evidence * cross_source * agreement * freshness)

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
