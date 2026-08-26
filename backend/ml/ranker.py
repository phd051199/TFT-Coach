from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from backend.app.catalog import Catalog
from backend.app.features import encode, make_feature_space


ROOT = Path(__file__).resolve().parents[2]


class LearnedRanker:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        configured = os.getenv("HEXCOACH_MODEL", "backend/models/board_ranker.keras")
        self.path = ROOT / configured
        self.meta_path = self.path.with_suffix(".meta.json")
        self.space = make_feature_space(catalog)
        self._model = None
        self.error: str | None = None
        self._cache: dict[tuple, float] = {}
        self._uncertainty_cache: dict[tuple, float] = {}
        self._metadata_cache: dict | None = None

    @property
    def available(self) -> bool:
        return self.path.exists()

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.path.exists():
            return None
        try:
            import tensorflow as tf

            member_paths = list(self.metadata().get("members") or [])
            if not member_paths:
                member_paths = [str(self.path.relative_to(ROOT))]
            models = []
            for member in member_paths:
                model = tf.keras.models.load_model(ROOT / member)
                expected = int(model.input_shape[-1])
                if expected != self.space.size:
                    self.error = f"feature-size mismatch: model={expected}, runtime={self.space.size}"
                    return None
                models.append(model)
            self._model = models
        except Exception as exc:
            self.error = str(exc)
            return None
        return self._model

    def score(
        self,
        units: list[str],
        traits: list[str] | None = None,
        items: list[str] | None = None,
        level: int = 8,
        sample_kind: str = "final_board",
    ) -> float | None:
        key = (
            tuple(sorted(units)),
            tuple(sorted(traits or [])),
            tuple(sorted(items or [])),
            int(level),
            sample_kind,
        )
        if key in self._cache:
            return self._cache[key]
        models = self._load()
        if models is None:
            return None
        vector = encode(
            self.catalog,
            self.space,
            units=units,
            traits=traits,
            items=items,
            level=level,
            sample_kind=sample_kind,
        )
        predictions = [float(model.predict(np.expand_dims(vector, 0), verbose=0)[0][0]) for model in models]
        value = self._calibrate(float(np.mean(predictions)))
        self._cache[key] = value
        self._uncertainty_cache[key] = float(np.std(predictions))
        return value

    def score_many(self, rows: list[dict]) -> list[float | None]:
        if not rows:
            return []
        models = self._load()
        if models is None:
            return [None] * len(rows)
        output: list[float | None] = [None] * len(rows)
        missing_indexes: list[int] = []
        vectors: list[np.ndarray] = []
        keys: list[tuple] = []
        for index, row in enumerate(rows):
            key = (
                tuple(sorted(row.get("units") or [])),
                tuple(sorted(row.get("traits") or [])),
                tuple(sorted(row.get("items") or [])),
                int(row.get("level") or 8),
                str(row.get("sample_kind") or "final_board"),
            )
            cached = self._cache.get(key)
            if cached is not None:
                output[index] = cached
                continue
            vectors.append(
                encode(
                    self.catalog,
                    self.space,
                    units=list(row.get("units") or []),
                    traits=list(row.get("traits") or []),
                    items=list(row.get("items") or []),
                    level=int(row.get("level") or 8),
                    sample_kind=str(row.get("sample_kind") or "final_board"),
                )
            )
            missing_indexes.append(index)
            keys.append(key)
        if vectors:
            member_predictions = np.stack([model.predict(np.stack(vectors), verbose=0).reshape(-1) for model in models])
            predictions = np.mean(member_predictions, axis=0)
            uncertainties = np.std(member_predictions, axis=0)
            for index, key, prediction, uncertainty in zip(missing_indexes, keys, predictions, uncertainties, strict=True):
                value = self._calibrate(float(prediction))
                self._cache[key] = value
                self._uncertainty_cache[key] = float(uncertainty)
                output[index] = value
        return output

    def uncertainty(
        self,
        units: list[str],
        traits: list[str] | None = None,
        items: list[str] | None = None,
        level: int = 8,
        sample_kind: str = "final_board",
    ) -> float | None:
        key = (
            tuple(sorted(units)),
            tuple(sorted(traits or [])),
            tuple(sorted(items or [])),
            int(level),
            sample_kind,
        )
        if key not in self._cache:
            self.score(units, traits, items, level, sample_kind)
        return self._uncertainty_cache.get(key)

    def metadata(self) -> dict:
        if self._metadata_cache is not None:
            return self._metadata_cache
        if not self.meta_path.exists():
            return {}
        try:
            self._metadata_cache = json.loads(self.meta_path.read_text("utf-8"))
            return self._metadata_cache
        except Exception:
            return {}

    def _calibrate(self, value: float) -> float:
        calibration = self.metadata().get("calibration") or {}
        slope = float(calibration.get("slope") or 1.0)
        intercept = float(calibration.get("intercept") or 0.0)
        return max(0.0, min(1.0, value * slope + intercept))


class ItemRanker(LearnedRanker):
    def __init__(self, catalog: Catalog) -> None:
        super().__init__(catalog)
        configured = os.getenv("HEXCOACH_ITEM_MODEL", "backend/models/item_ranker.keras")
        self.path = ROOT / configured
        self.meta_path = self.path.with_suffix(".meta.json")
        self._metadata_cache = None

    def score_build(self, unit_id: str, item_ids: list[str]) -> float | None:
        return self.score(
            units=[unit_id],
            items=item_ids,
            level=8,
            sample_kind="item_holder_build",
        )
