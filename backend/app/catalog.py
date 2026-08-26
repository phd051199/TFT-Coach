from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SET_PATH = ROOT / "src" / "data" / "set18.generated.json"
META_PATH = ROOT / "src" / "data" / "meta.generated.json"


@dataclass(frozen=True)
class Catalog:
    raw: dict[str, Any]
    meta: dict[str, Any]
    champions: list[dict[str, Any]]
    traits: list[dict[str, Any]]
    items: list[dict[str, Any]]
    champion_by_id: dict[str, dict[str, Any]]
    champion_by_name: dict[str, dict[str, Any]]
    trait_by_id: dict[str, dict[str, Any]]
    item_by_id: dict[str, dict[str, Any]]


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    raw = json.loads(SET_PATH.read_text("utf-8"))
    meta = json.loads(META_PATH.read_text("utf-8"))
    champions = raw["champions"]
    traits = raw["traits"]
    items = raw["items"]
    return Catalog(
        raw=raw,
        meta=meta,
        champions=champions,
        traits=traits,
        items=items,
        champion_by_id={champion["id"]: champion for champion in champions},
        champion_by_name={champion["name"]: champion for champion in champions},
        trait_by_id={trait["id"]: trait for trait in traits},
        item_by_id={item["id"]: item for item in items},
    )
