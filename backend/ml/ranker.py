from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from backend.app.catalog import Catalog
from backend.app.features import encode, make_feature_space
from backend.ml.position import CELL_COUNT, encode_position, make_position_space
from backend.ml.stars import encode_star, make_star_space


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

    def warmup(self) -> None:
        self._load()

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


class ItemAffinityRanker(LearnedRanker):
    """Predict whether one specific item naturally belongs on one specific champion.

    This model is deliberately separate from the 2/3-item build ranker. Single-item holder
    affinity is trained from MetaTFT's holder-level place-change statistics, while the build
    ranker keeps learning full build strength. Mixing both targets made holder selection less
    stable because they encode different questions.
    """

    def __init__(self, catalog: Catalog) -> None:
        super().__init__(catalog)
        configured = os.getenv("HEXCOACH_ITEM_AFFINITY_MODEL", "backend/models/item_affinity_ranker.keras")
        self.path = ROOT / configured
        self.meta_path = self.path.with_suffix(".meta.json")
        self._metadata_cache = None

    def score_affinity(self, unit_id: str, item_id: str) -> float | None:
        return self.score(
            units=[unit_id],
            items=[item_id],
            level=8,
            sample_kind="item_affinity",
        )


class ItemPairRanker(LearnedRanker):
    """Predict how naturally two completed items are paired on the same holder.

    Unlike ItemRanker, which predicts placement strength of an observed full build, this
    model is trained directly on item co-occurrence. That gives the optimizer an explicit
    signal for pairs that are individually good but rarely/never built together.
    """

    def __init__(self, catalog: Catalog) -> None:
        super().__init__(catalog)
        configured = os.getenv("HEXCOACH_ITEM_PAIR_MODEL", "backend/models/item_pair_ranker.keras")
        self.path = ROOT / configured
        self.meta_path = self.path.with_suffix(".meta.json")
        self._metadata_cache = None

    def score_pair(self, unit_id: str, left_item_id: str, right_item_id: str) -> float | None:
        return self.score(
            units=[unit_id],
            items=sorted([left_item_id, right_item_id]),
            level=8,
            sample_kind="item_pair_affinity",
        )


class RerollRanker(LearnedRanker):
    def __init__(self, catalog: Catalog) -> None:
        super().__init__(catalog)
        configured = os.getenv("HEXCOACH_REROLL_MODEL", "backend/models/reroll_ranker.keras")
        self.path = ROOT / configured
        self.meta_path = self.path.with_suffix(".meta.json")
        self._metadata_cache = None

    def score_board(self, units: list[str], level: int) -> float | None:
        return self.score(units=units, level=level, sample_kind="reroll_board")


class StarRanker:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        configured = os.getenv("HEXCOACH_STAR_MODEL", "backend/models/star_ranker.keras")
        self.path = ROOT / configured
        self.meta_path = self.path.with_suffix(".meta.json")
        self.space = make_star_space(catalog)
        self._models = None
        self.error: str | None = None
        self._cache: dict[tuple[tuple[str, ...], str, int], tuple[np.ndarray, float]] = {}
        self._metadata_cache: dict | None = None
        self._priors_cache: dict | None = None

    @property
    def available(self) -> bool:
        return self.path.exists()

    def warmup(self) -> None:
        self._load()

    def _load(self):
        if self._models is not None:
            return self._models
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
            self._models = models
        except Exception as exc:
            self.error = str(exc)
            return None
        return self._models

    def predict_many(self, rows: list[dict]) -> list[dict | None]:
        if not rows:
            return []
        models = self._load()
        if models is None:
            return [None] * len(rows)
        output: list[dict | None] = [None] * len(rows)
        missing_indexes: list[int] = []
        vectors: list[np.ndarray] = []
        keys: list[tuple[tuple[str, ...], str, int]] = []
        for index, row in enumerate(rows):
            board = list(row.get("board") or [])
            unit = str(row.get("unit") or "")
            level = int(row.get("level") or len(board) or 8)
            key = (tuple(sorted(board)), unit, level)
            cached = self._cache.get(key)
            if cached is not None:
                distribution, uncertainty = cached
                star = int(np.argmax(distribution)) + 1
                output[index] = {
                    "stars": star,
                    "confidence": float(distribution[star - 1]),
                    "distribution": distribution.tolist(),
                    "uncertainty": uncertainty,
                }
                continue
            vectors.append(encode_star(self.catalog, self.space, board, unit, level))
            missing_indexes.append(index)
            keys.append(key)
        if vectors:
            member_predictions = np.stack([model.predict(np.stack(vectors), verbose=0) for model in models])
            mean_predictions = np.mean(member_predictions, axis=0)
            std_predictions = np.std(member_predictions, axis=0)
            for output_index, key, distribution, std in zip(
                missing_indexes, keys, mean_predictions, std_predictions, strict=True
            ):
                distribution = self._blend_empirical_prior(key[1], distribution)
                uncertainty = float(np.mean(std))
                self._cache[key] = (distribution, uncertainty)
                star = int(np.argmax(distribution)) + 1
                output[output_index] = {
                    "stars": star,
                    "confidence": float(distribution[star - 1]),
                    "distribution": distribution.tolist(),
                    "uncertainty": uncertainty,
                }
        return output

    def _priors(self) -> dict:
        if self._priors_cache is not None:
            return self._priors_cache
        configured = str(self.metadata().get("priorFile") or "backend/models/star-priors.json")
        try:
            self._priors_cache = json.loads((ROOT / configured).read_text("utf-8"))
        except Exception:
            self._priors_cache = {}
        return self._priors_cache

    def _blend_empirical_prior(self, unit_id: str, model_distribution: np.ndarray) -> np.ndarray:
        priors = self._priors()
        champion = self.catalog.champion_by_id.get(unit_id, {})
        cost = int(champion.get("cost") or 1)
        cost_row = (priors.get("costs") or {}).get(str(cost)) or {}
        unit_row = (priors.get("units") or {}).get(unit_id) or {}
        cost_distribution = np.asarray(cost_row.get("distribution") or [0.25, 0.70, 0.05], dtype=np.float32)
        unit_distribution = np.asarray(unit_row.get("distribution") or cost_distribution, dtype=np.float32)
        games = max(0, int(unit_row.get("games") or 0))
        # Smooth sparse per-unit observations toward the same-cost population first.
        smoothed_prior = (unit_distribution * games + cost_distribution * 8.0) / max(1.0, games + 8.0)
        empirical_weight = min(0.68, np.log1p(games) / 5.5) if games > 0 else 0.18
        blended = model_distribution * (1.0 - empirical_weight) + smoothed_prior * empirical_weight
        return blended / max(1e-8, float(np.sum(blended)))

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


class PositionRanker:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        configured = os.getenv("HEXCOACH_POSITION_MODEL", "backend/models/position_ranker.keras")
        self.path = ROOT / configured
        self.meta_path = self.path.with_suffix(".meta.json")
        self.space = make_position_space(catalog)
        self._models = None
        self.error: str | None = None
        self._cache: dict[tuple[tuple[str, ...], str], tuple[np.ndarray, np.ndarray]] = {}
        self._metadata_cache: dict | None = None

    @property
    def available(self) -> bool:
        return self.path.exists()

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

    def _load(self):
        if self._models is not None:
            return self._models
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
                if expected != self.space.size or int(model.output_shape[-1]) != CELL_COUNT:
                    self.error = (
                        f"position-model shape mismatch: input={expected}/{self.space.size}, "
                        f"output={model.output_shape[-1]}/{CELL_COUNT}"
                    )
                    return None
                models.append(model)
            self._models = models
        except Exception as exc:
            self.error = str(exc)
            return None
        return self._models

    def warmup(self) -> None:
        self._load()

    def predict_many(self, rows: list[tuple[list[str], str]]) -> list[tuple[np.ndarray, np.ndarray] | None]:
        if not rows:
            return []
        models = self._load()
        if models is None:
            return [None] * len(rows)
        output: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(rows)
        missing: list[int] = []
        vectors: list[np.ndarray] = []
        keys: list[tuple[tuple[str, ...], str]] = []
        for index, (board_units, unit_id) in enumerate(rows):
            key = (tuple(sorted(board_units)), unit_id)
            cached = self._cache.get(key)
            if cached is not None:
                output[index] = cached
                continue
            missing.append(index)
            keys.append(key)
            vectors.append(encode_position(self.catalog, self.space, board_units, unit_id))
        if vectors:
            batch = np.stack(vectors)
            member_predictions = np.stack([
                np.asarray(model(batch, training=False))
                for model in models
            ])
            means = np.mean(member_predictions, axis=0)
            stds = np.std(member_predictions, axis=0)
            for index, key, mean, std in zip(missing, keys, means, stds, strict=True):
                mean = np.asarray(mean, dtype=np.float32)
                total = float(mean.sum())
                if total > 0:
                    mean = mean / total
                value = (mean, np.asarray(std, dtype=np.float32))
                self._cache[key] = value
                output[index] = value
        return output

    def predict(self, board_units: list[str], unit_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        return self.predict_many([(board_units, unit_id)])[0]
