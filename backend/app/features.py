from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .catalog import Catalog


@dataclass(frozen=True)
class FeatureSpace:
    champion_ids: tuple[str, ...]
    trait_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    champion_index: dict[str, int]
    trait_index: dict[str, int]
    item_index: dict[str, int]
    dense_size: int

    @property
    def size(self) -> int:
        return len(self.champion_ids) + len(self.trait_ids) + len(self.item_ids) + self.dense_size


def make_feature_space(catalog: Catalog) -> FeatureSpace:
    champion_ids = tuple(champion["id"] for champion in catalog.champions)
    trait_ids = tuple(trait["id"] for trait in catalog.traits if trait.get("searchable", True))
    item_ids = tuple(item["id"] for item in catalog.items if item.get("category") in {"completed", "artifact", "radiant", "emblem"})
    return FeatureSpace(
        champion_ids=champion_ids,
        trait_ids=trait_ids,
        item_ids=item_ids,
        champion_index={value: index for index, value in enumerate(champion_ids)},
        trait_index={value: index for index, value in enumerate(trait_ids)},
        item_index={value: index for index, value in enumerate(item_ids)},
        dense_size=12,
    )


def encode(
    catalog: Catalog,
    space: FeatureSpace,
    units: list[str],
    traits: list[str] | None = None,
    items: list[str] | None = None,
    level: int = 8,
    sample_kind: str = "final_board",
) -> np.ndarray:
    vector = np.zeros(space.size, dtype=np.float32)
    unit_offset = 0
    trait_offset = len(space.champion_ids)
    item_offset = trait_offset + len(space.trait_ids)
    dense_offset = item_offset + len(space.item_ids)

    board: list[dict[str, Any]] = []
    for unit_id in units:
        champion = catalog.champion_by_id.get(unit_id)
        if champion:
            board.append(champion)
        index = space.champion_index.get(unit_id)
        if index is not None:
            vector[unit_offset + index] = 1.0

    inferred_traits: dict[str, int] = {}
    for champion in board:
        for trait_id in champion.get("traits", []):
            inferred_traits[trait_id] = inferred_traits.get(trait_id, 0) + 1
    for trait_name in traits or []:
        # Aggregate sources may append _1/_2 to the base trait id.
        trait_id = trait_name
        if trait_id not in catalog.trait_by_id:
            parts = trait_name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit() and parts[0] in catalog.trait_by_id:
                trait_id = parts[0]
        inferred_traits.setdefault(trait_id, 1)

    active_count = 0
    for trait_id, count in inferred_traits.items():
        index = space.trait_index.get(trait_id)
        if index is None:
            continue
        vector[trait_offset + index] = min(10, count) / 10.0
        trait = catalog.trait_by_id.get(trait_id, {})
        if any(int(bp) <= count for bp in trait.get("breakpoints", [])):
            active_count += 1

    for item_id in items or []:
        index = space.item_index.get(item_id)
        if index is not None:
            vector[item_offset + index] = min(3.0, vector[item_offset + index] + 1.0)

    costs = [int(champion.get("cost", 1)) for champion in board]
    roles = [str(champion.get("role", "")) for champion in board]
    frontline = sum("Tank" in role or "Fighter" in role for role in roles)
    ad = sum(role.startswith("AD") for role in roles)
    ap = sum(role.startswith("AP") for role in roles)
    dense = np.asarray(
        [
            min(10, level) / 10.0,
            len(board) / 10.0,
            (sum(costs) / max(1, len(costs))) / 5.0,
            sum(cost == 1 for cost in costs) / max(1, len(costs)),
            sum(cost == 2 for cost in costs) / max(1, len(costs)),
            sum(cost == 3 for cost in costs) / max(1, len(costs)),
            sum(cost >= 4 for cost in costs) / max(1, len(costs)),
            active_count / 10.0,
            frontline / max(1, len(board)),
            (ad - ap) / max(1, len(board)),
            1.0 if sample_kind == "early_board" else 0.0,
            1.0 if sample_kind == "item_holder_build" else 0.0,
        ],
        dtype=np.float32,
    )
    vector[dense_offset : dense_offset + len(dense)] = dense
    return vector
