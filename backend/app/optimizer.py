from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from backend.ml.position import CELL_COUNT, cell_name, display_grid_position
from backend.ml.ranker import (
    ItemAffinityRanker,
    ItemPairRanker,
    ItemRanker,
    LearnedRanker,
    PositionRanker,
    RerollRanker,
    StarRanker,
)

from .catalog import Catalog, load_catalog


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = ROOT / "backend" / "data" / "metatft.snapshot.json"
OPGG_PATH = ROOT / "backend" / "data" / "opgg-live.snapshot.json"
HIGH_ELO_PATH = ROOT / "backend" / "data" / "high-elo-priors.json"
HEALTH_PATH = ROOT / "backend" / "data" / "source-health.json"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def placement_strength(avg: float) -> float:
    if not (1.0 <= avg <= 8.0):
        return 0.5
    return clamp((8.5 - avg) / 7.5)


def evidence(games: int) -> float:
    return clamp(math.log1p(max(0, games)) / math.log(501.0))


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


class HybridCoach:
    """Candidate generator + empirical/live stats + TensorFlow ranker.

    The model never invents a board. We rank boards observed in current/live statistics and
    fall back to a trait-aware beam search when no observed opener matches the user's level.
    """

    def __init__(self, catalog: Catalog | None = None) -> None:
        self.catalog = catalog or load_catalog()
        self.board_ranker = LearnedRanker(self.catalog)
        self.item_ranker = ItemRanker(self.catalog)
        self.item_affinity_ranker = ItemAffinityRanker(self.catalog)
        self.item_pair_ranker = ItemPairRanker(self.catalog)
        self.position_ranker = PositionRanker(self.catalog)
        self.reroll_ranker = RerollRanker(self.catalog)
        self.star_ranker = StarRanker(self.catalog)
        self.snapshot = self._load_json(SNAPSHOT_PATH, {"clusters": []})
        self.opgg = self._load_json(OPGG_PATH, {})
        self.high_elo = self._load_json(HIGH_ELO_PATH, {"units": {}, "itemHolders": {}})
        self.health = self._load_json(HEALTH_PATH, {"sources": {}, "total": 0})
        self.clusters: list[dict[str, Any]] = list(self.snapshot.get("clusters") or [])
        self._best_final_cache: dict[str, tuple[list[str], float, dict[str, Any] | None, int]] = {}
        self._level_board_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._item_role_cache: dict[tuple[str, str], float] = {}
        self._item_stat_cache: dict[tuple[str, str, str], tuple[float, int, float]] = {}
        self._item_affinity_scores: dict[tuple[str, str], float] = {}
        self._item_pair_score_cache: dict[tuple[str, str, str], float] = {}
        self._emblem_trait_cache: dict[str, str | None] = {}
        self._position_distribution_cache: dict[tuple[str, tuple[str, ...], str], tuple[list[float], int, float]] = {}
        self._index_opgg()
        self._index_item_pairs()
        self._precompute_ml()
        self.item_affinity_ranker.warmup()
        self.item_pair_ranker.warmup()
        self.position_ranker.warmup()
        self.reroll_ranker.warmup()
        self.star_ranker.warmup()

    @staticmethod
    def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            return default

    def reload(self) -> None:
        self.snapshot = self._load_json(SNAPSHOT_PATH, {"clusters": []})
        self.opgg = self._load_json(OPGG_PATH, {})
        self.high_elo = self._load_json(HIGH_ELO_PATH, {"units": {}, "itemHolders": {}})
        self.health = self._load_json(HEALTH_PATH, {"sources": {}, "total": 0})
        self.clusters = list(self.snapshot.get("clusters") or [])
        self._best_final_cache = {}
        self._level_board_cache = {}
        self._item_role_cache = {}
        self._item_stat_cache = {}
        self._item_affinity_scores = {}
        self._item_pair_score_cache = {}
        self._emblem_trait_cache = {}
        self._position_distribution_cache = {}
        self._index_opgg()
        self._index_item_pairs()
        self.board_ranker = LearnedRanker(self.catalog)
        self.item_ranker = ItemRanker(self.catalog)
        self.item_affinity_ranker = ItemAffinityRanker(self.catalog)
        self.item_pair_ranker = ItemPairRanker(self.catalog)
        self.position_ranker = PositionRanker(self.catalog)
        self.reroll_ranker = RerollRanker(self.catalog)
        self.star_ranker = StarRanker(self.catalog)
        self._precompute_ml()
        self.item_affinity_ranker.warmup()
        self.item_pair_ranker.warmup()
        self.position_ranker.warmup()
        self.reroll_ranker.warmup()
        self.star_ranker.warmup()

    def _index_opgg(self) -> None:
        self.opgg_units = {
            str(row.get("unit")): row for row in self.opgg.get("unitStats") or [] if row.get("unit")
        }
        self.opgg_items = {
            str(row.get("item")): row for row in self.opgg.get("itemStats") or [] if row.get("item")
        }
        self.opgg_holders = {
            (str(row.get("item")), str(row.get("unit"))): row
            for row in self.opgg.get("itemHolders") or []
            if row.get("item") and row.get("unit")
        }
        self.opgg_comps = list(self.opgg.get("comps") or [])
        self.opgg_unit_percentile = self._relative_strength_index(list(self.opgg_units.values()), "unit")
        self.opgg_item_percentile = self._relative_strength_index(list(self.opgg_items.values()), "item")
        self.opgg_holder_percentile = self._relative_strength_index(
            list(self.opgg_holders.values()),
            lambda row: f"{row.get('item')}::{row.get('unit')}",
        )
        self.opgg_comp_percentile = self._relative_strength_index(
            self.opgg_comps,
            lambda row: "|".join(sorted(str(value) for value in row.get("units") or [])),
        )

    def _index_item_pairs(self) -> None:
        """Index weighted item co-occurrence from observed 2/3-item holder builds."""
        self._pair_global_items: Counter[str] = Counter()
        self._pair_global_pairs: Counter[tuple[str, str]] = Counter()
        self._pair_holder_items: dict[str, Counter[str]] = {}
        self._pair_holder_pairs: dict[str, Counter[tuple[str, str]]] = {}
        for cluster in self.clusters:
            for build in cluster.get("builds") or []:
                holder_id = str(build.get("unit") or "")
                count = int(build.get("count") or 0)
                items = sorted(set(str(value) for value in build.get("items") or [] if value))
                if not holder_id or count <= 0 or len(items) < 2:
                    continue
                holder_items = self._pair_holder_items.setdefault(holder_id, Counter())
                holder_pairs = self._pair_holder_pairs.setdefault(holder_id, Counter())
                for item_id in items:
                    self._pair_global_items[item_id] += count
                    holder_items[item_id] += count
                for left_index, left in enumerate(items):
                    for right in items[left_index + 1 :]:
                        pair = (left, right)
                        self._pair_global_pairs[pair] += count
                        holder_pairs[pair] += count

    def _relative_strength_index(self, rows: list[dict[str, Any]], key: str | Any) -> dict[str, float]:
        """Normalize a source to within-source percentiles to avoid cross-site scale drift."""
        scored: list[tuple[float, str]] = []
        for row in rows:
            if callable(key):
                row_key = str(key(row))
            else:
                row_key = str(row.get(key) or "")
            if not row_key:
                continue
            score = self._aggregate_strength(float(row.get("avg") or 0), row.get("top4"), row.get("win"))
            scored.append((score, row_key))
        scored.sort(key=lambda value: value[0])
        if not scored:
            return {}
        denominator = max(1, len(scored) - 1)
        return {
            row_key: 0.32 + (index / denominator) * 0.44
            for index, (_, row_key) in enumerate(scored)
        }

    @staticmethod
    def _ranker_value_reliability(ranker: LearnedRanker) -> float:
        """Estimate how much an absolute neural score should influence runtime decisions.

        Ranking models can still order examples well while being badly shifted on an
        independent source. Absolute score blending (core-item quality, board strength) must
        care about that calibration error, otherwise one overfit model can overpower live
        empirical data. Keep a non-zero floor because the model is still useful as a prior.
        """
        metadata = ranker.metadata()
        evaluation = metadata.get("evaluation") or {}
        internal_gain = float(evaluation.get("improvementVsBaseline") or 0.0)
        internal_quality = clamp(0.48 + internal_gain * 1.35, 0.25, 0.95)
        external = metadata.get("externalEvaluation") or {}
        samples = int(external.get("samples") or 0)
        baseline = float(external.get("baselineMAE") or 0.0)
        mae = float(external.get("mae") or 0.0)
        if samples < 8 or baseline <= 0 or mae <= 0:
            return internal_quality
        mae_quality = clamp(baseline / mae, 0.15, 1.0)
        ranking_accuracy = float(external.get("rankingAccuracy") or 0.5)
        ranking_quality = clamp((ranking_accuracy - 0.5) / 0.25)
        external_quality = mae_quality * 0.75 + ranking_quality * 0.25
        return clamp(internal_quality * external_quality, 0.15, 0.95)

    @staticmethod
    def _aggregate_strength(avg: float, top4: float | None = None, win: float | None = None) -> float:
        values: list[tuple[float, float]] = [(placement_strength(avg), 0.72)]
        if top4 is not None:
            values.append((clamp(float(top4)), 0.20))
        if win is not None:
            values.append((clamp(float(win)), 0.08))
        denominator = sum(weight for _, weight in values)
        return sum(value * weight for value, weight in values) / denominator

    def _opgg_unit_strength(self, unit_id: str) -> tuple[float, float]:
        row = self.opgg_units.get(unit_id)
        if not row:
            return 0.5, 0.0
        return (
            self.opgg_unit_percentile.get(unit_id, 0.5),
            evidence(int(row.get("games") or 0)),
        )

    def _opgg_item_strength(self, item_id: str, holder_id: str | None = None) -> tuple[float, float]:
        row = self.opgg_holders.get((item_id, holder_id)) if holder_id else self.opgg_items.get(item_id)
        if not row:
            return 0.5, 0.0
        relative_key = f"{item_id}::{holder_id}" if holder_id else item_id
        relative = self.opgg_holder_percentile.get(relative_key, 0.5) if holder_id else self.opgg_item_percentile.get(relative_key, 0.5)
        return (
            relative,
            evidence(int(row.get("games") or 0)),
        )

    def _high_elo_unit_prior(self, unit_id: str) -> tuple[float, float]:
        row = (self.high_elo.get("units") or {}).get(unit_id)
        if not row:
            return 0.5, 0.0
        games = int(row.get("games") or 0)
        return (
            self._aggregate_strength(float(row.get("avgPlacement") or 0), row.get("top4Rate"), row.get("winRate")),
            evidence(games),
        )

    def _high_elo_item_holder_prior(self, item_id: str, holder_id: str) -> tuple[float, float]:
        row = (self.high_elo.get("itemHolders") or {}).get(f"{item_id}::{holder_id}")
        if not row:
            return 0.5, 0.0
        games = int(row.get("games") or 0)
        return (
            self._aggregate_strength(float(row.get("avgPlacement") or 0), row.get("top4Rate"), row.get("winRate")),
            evidence(games),
        )

    def _high_elo_board_prior(self, unit_ids: list[str]) -> tuple[float, float]:
        signals = [self._high_elo_unit_prior(unit_id) for unit_id in unit_ids]
        signals = [(strength, ev) for strength, ev in signals if ev > 0]
        if not signals:
            return 0.5, 0.0
        evidence_sum = sum(ev for _, ev in signals)
        strength = sum(value * ev for value, ev in signals) / max(1e-8, evidence_sum)
        coverage = len(signals) / max(1, len(unit_ids))
        return strength, clamp((evidence_sum / len(signals)) * coverage)

    def _opgg_comp_match(self, unit_ids: list[str]) -> tuple[float, float, float, dict[str, Any] | None]:
        target = set(unit_ids)
        if not target:
            return 0.5, 0.0, 0.0, None
        best: tuple[float, float, float, dict[str, Any]] | None = None
        for row in self.opgg_comps:
            other = set(self._known_units(list(row.get("units") or [])))
            if not other:
                continue
            intersection = len(target & other)
            similarity = intersection / max(len(target), len(other))
            if similarity < 0.55:
                continue
            signature = "|".join(sorted(str(value) for value in row.get("units") or []))
            strength = self.opgg_comp_percentile.get(signature, 0.5)
            ev = evidence(int(row.get("games") or 0))
            candidate = (similarity * (0.75 + ev * 0.25), strength, ev, row)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            return 0.5, 0.0, 0.0, None
        return best[1], best[2], min(1.0, best[0]), best[3]

    def _known_units(self, values: list[str]) -> list[str]:
        return unique([value for value in values if value in self.catalog.champion_by_id])

    def _precompute_ml(self) -> None:
        rows: list[dict[str, Any]] = []
        destinations: list[dict[str, Any]] = []
        board_model_reliability = self._ranker_value_reliability(self.board_ranker)
        for cluster in self.clusters:
            for level_key, options in (cluster.get("options") or {}).items():
                try:
                    level = int(level_key)
                except ValueError:
                    continue
                for option in options:
                    units = self._known_units(list(option.get("units") or []))
                    if len(units) < 3:
                        continue
                    rows.append({
                        "units": units,
                        "traits": list(option.get("traits") or []),
                        "level": level,
                        "sample_kind": "exact_comp_option",
                    })
                    destinations.append(option)
            for level_key, options in (cluster.get("earlyOptions") or {}).items():
                try:
                    level = int(level_key)
                except ValueError:
                    continue
                for option in options:
                    units = self._known_units(list(option.get("units") or []))
                    if len(units) < 3:
                        continue
                    rows.append({"units": units, "level": level, "sample_kind": "early_board"})
                    destinations.append(option)
        for destination, score in zip(destinations, self.board_ranker.score_many(rows), strict=True):
            if score is not None:
                observed = placement_strength(float(destination.get("avg") or 0))
                destination["ml"] = observed * (1.0 - board_model_reliability) + score * board_model_reliability
                destination["mlReliability"] = board_model_reliability
        for destination, row in zip(destinations, rows, strict=True):
            uncertainty = self.board_ranker.uncertainty(
                units=list(row.get("units") or []),
                traits=list(row.get("traits") or []),
                level=int(row.get("level") or 8),
                sample_kind=str(row.get("sample_kind") or "final_board"),
            )
            if uncertainty is not None:
                destination["mlStd"] = uncertainty

        item_rows: list[dict[str, Any]] = []
        item_destinations: list[dict[str, Any]] = []
        item_model_reliability = self._ranker_value_reliability(self.item_ranker)
        for cluster in self.clusters:
            for build in cluster.get("builds") or []:
                unit_id = str(build.get("unit") or "")
                items = [str(value) for value in build.get("items") or [] if value]
                if unit_id not in self.catalog.champion_by_id or not items:
                    continue
                item_rows.append({
                    "units": [unit_id],
                    "items": items,
                    "sample_kind": "item_holder_build",
                    "level": 8,
                })
                item_destinations.append(build)
        for destination, score in zip(item_destinations, self.item_ranker.score_many(item_rows), strict=True):
            if score is not None:
                observed = placement_strength(float(destination.get("avg") or 0))
                destination["ml"] = observed * (1.0 - item_model_reliability) + score * item_model_reliability
                destination["mlReliability"] = item_model_reliability
        for destination, row in zip(item_destinations, item_rows, strict=True):
            uncertainty = self.item_ranker.uncertainty(
                units=list(row["units"]),
                items=list(row["items"]),
                level=8,
                sample_kind="item_holder_build",
            )
            if uncertainty is not None:
                destination["mlStd"] = uncertainty

        # Single-item holder affinity is reused across every comp. Batch all champion/item
        # pairs once at startup so online item solving never triggers thousands of tiny
        # TensorFlow calls while the user changes components.
        affinity_rows: list[dict[str, Any]] = []
        affinity_keys: list[tuple[str, str]] = []
        affinity_items = [
            item for item in self.catalog.items
            if item.get("category") in {"completed", "emblem"}
        ]
        for champion in self.catalog.champions:
            unit_id = str(champion["id"])
            for item in affinity_items:
                item_id = str(item["id"])
                affinity_rows.append({
                    "units": [unit_id],
                    "items": [item_id],
                    "sample_kind": "item_affinity",
                    "level": 8,
                })
                affinity_keys.append((unit_id, item_id))
        for key, score in zip(affinity_keys, self.item_affinity_ranker.score_many(affinity_rows), strict=True):
            if score is not None:
                self._item_affinity_scores[key] = score

        # Pair compatibility is also a request-hot signal. Batch only item pairs that have
        # meaningful holder support, which covers normal carry/tank bundles while avoiding a
        # champion × every-item-pair Cartesian explosion. Sparse/unseen pairs stay neutral at
        # runtime unless empirical co-occurrence or a non-stacking effect gives evidence.
        pair_rows: list[dict[str, Any]] = []
        pair_keys: list[tuple[str, str, str]] = []
        for holder_id, support in self._pair_holder_items.items():
            pool = [item_id for item_id, count in support.most_common(18) if count >= 5]
            for left_index, left in enumerate(sorted(pool)):
                for right in sorted(pool)[left_index + 1 :]:
                    pair_rows.append({
                        "units": [holder_id],
                        "items": [left, right],
                        "sample_kind": "item_pair_affinity",
                        "level": 8,
                    })
                    pair_keys.append((holder_id, left, right))
        for key, score in zip(pair_keys, self.item_pair_ranker.score_many(pair_rows), strict=True):
            if score is not None:
                self._item_pair_score_cache[key] = score

    def _entity_name(self, entity_id: str) -> str | None:
        champion = self.catalog.champion_by_id.get(entity_id)
        if champion:
            return str(champion["name"])
        trait = self.catalog.trait_by_id.get(entity_id)
        if trait:
            return str(trait["name"])
        return None

    def comp_name(self, cluster: dict[str, Any]) -> str:
        labels: list[str] = []
        for part in cluster.get("nameParts") or []:
            entity_id = str(part.get("name") or "") if isinstance(part, dict) else str(part)
            translated = self._entity_name(entity_id)
            if translated and translated not in labels:
                labels.append(translated)
        if not labels:
            for unit_id in self._known_units(list(cluster.get("centroidUnits") or []))[:2]:
                labels.append(str(self.catalog.champion_by_id[unit_id]["name"]))
        return " · ".join(labels[:3]) or f"Đội hình {cluster.get('id', '?')}"

    def active_traits(self, unit_ids: list[str], extra_trait_ids: list[str] | None = None) -> list[dict[str, Any]]:
        counts: Counter[str] = Counter()
        for unit_id in unit_ids:
            champion = self.catalog.champion_by_id.get(unit_id)
            if champion:
                counts.update(champion.get("traits") or [])
        counts.update(value for value in (extra_trait_ids or []) if value in self.catalog.trait_by_id)
        output: list[dict[str, Any]] = []
        for trait_id, count in counts.items():
            trait = self.catalog.trait_by_id.get(trait_id)
            if not trait:
                continue
            breakpoints = sorted(int(value) for value in trait.get("breakpoints") or [])
            active = max((value for value in breakpoints if value <= count), default=0)
            nxt = next((value for value in breakpoints if value > count), None)
            row = {"traitId": trait_id, "count": count, "activeBreakpoint": active}
            if nxt is not None:
                row["nextBreakpoint"] = nxt
            output.append(row)
        output.sort(
            key=lambda value: (value["activeBreakpoint"] > 0, value["activeBreakpoint"], value["count"]),
            reverse=True,
        )
        return output

    def _heuristic_board_score(self, unit_ids: list[str], owned: set[str], early: bool) -> float:
        units = [self.catalog.champion_by_id[value] for value in unit_ids if value in self.catalog.champion_by_id]
        traits = self.active_traits(unit_ids)
        trait_score = sum(1.0 + row["activeBreakpoint"] * 0.15 for row in traits if row["activeBreakpoint"])
        tanks = sum("Tank" in str(unit.get("role", "")) or "Fighter" in str(unit.get("role", "")) for unit in units)
        carries = sum(any(token in str(unit.get("role", "")) for token in ("Carry", "Caster", "Reaper", "Specialist")) for unit in units)
        own = sum(unit["id"] in owned for unit in units)
        raw = trait_score * 1.25 + min(3, tanks) * 0.85 + min(3, carries) * 0.7 + own * 1.5
        if early:
            raw -= sum(max(0, int(unit.get("cost") or 1) - 2) * 0.32 for unit in units)
        heuristic = clamp(raw / 12.5)
        live = [self._opgg_unit_strength(str(unit["id"])) for unit in units]
        live = [(strength, ev) for strength, ev in live if ev > 0]
        if not live:
            return heuristic
        live_mean = sum(strength * ev for strength, ev in live) / max(1e-8, sum(ev for _, ev in live))
        live_weight = 0.24 if early else 0.15
        return heuristic * (1.0 - live_weight) + live_mean * live_weight

    def _fallback_early_board(self, level: int, owned: set[str]) -> list[str]:
        size = max(3, min(10, level))
        pool = [
            champion for champion in self.catalog.champions
            if int(champion.get("cost") or 1) <= 3 or champion["id"] in owned
        ]
        seeds = [value for value in owned if value in self.catalog.champion_by_id]
        seeds.sort(key=lambda value: int(self.catalog.champion_by_id[value].get("cost") or 1))
        beam: list[list[str]] = [seeds[:size]] if seeds else [[]]
        width = 100
        while beam and len(beam[0]) < size:
            next_states: dict[tuple[str, ...], tuple[float, list[str]]] = {}
            for board in beam:
                used = set(board)
                for champion in pool:
                    unit_id = str(champion["id"])
                    if unit_id in used:
                        continue
                    candidate = board + [unit_id]
                    key = tuple(sorted(candidate))
                    score = self._heuristic_board_score(candidate, owned, True)
                    old = next_states.get(key)
                    if old is None or score > old[0]:
                        next_states[key] = (score, candidate)
            beam = [entry[1] for entry in sorted(next_states.values(), key=lambda value: value[0], reverse=True)[:width]]
        return beam[0] if beam else seeds[:size]

    def _best_early(self, cluster: dict[str, Any], level: int, owned: set[str]) -> tuple[list[str], float, dict[str, Any] | None]:
        early = cluster.get("earlyOptions") or {}
        available: list[int] = []
        for key in early:
            try:
                available.append(int(key))
            except ValueError:
                pass
        if not available:
            return [], 0.0, None
        nearest = sorted(available, key=lambda value: (abs(value - level), value > level, value))[:2]
        best: tuple[float, list[str], dict[str, Any]] | None = None
        for key in nearest:
            for option in early.get(str(key)) or []:
                units = self._known_units(list(option.get("units") or []))
                if len(units) < 3:
                    continue
                owned_hits = len(owned.intersection(units))
                overlap = owned_hits / max(1, len(owned)) if owned else 0.5
                observed = placement_strength(float(option.get("avg") or 0))
                learned = float(option.get("ml") if option.get("ml") is not None else self._heuristic_board_score(units, owned, True))
                sample = evidence(int(option.get("count") or 0))
                exact_level = clamp(1.0 - abs(len(units) - level) * 0.12)
                score = overlap * 0.35 + observed * 0.22 + learned * 0.24 + sample * 0.11 + exact_level * 0.08
                if best is None or score > best[0]:
                    best = (score, units, option)
        return (best[1], best[0], best[2]) if best else ([], 0.0, None)

    def _best_final(self, cluster: dict[str, Any]) -> tuple[list[str], float, dict[str, Any] | None, int]:
        cache_key = str(cluster.get("id") or id(cluster))
        cached = self._best_final_cache.get(cache_key)
        if cached is not None:
            return cached
        best: tuple[float, list[str], dict[str, Any], int] | None = None
        for key, rows in (cluster.get("options") or {}).items():
            try:
                level = int(key)
            except ValueError:
                continue
            if level < 6:
                continue
            for option in rows:
                units = self._known_units(list(option.get("units") or []))
                if len(units) < 4:
                    continue
                observed = placement_strength(float(option.get("avg") or 0))
                learned = float(option.get("ml") if option.get("ml") is not None else self._heuristic_board_score(units, set(), False))
                sample = evidence(int(option.get("count") or 0))
                score = observed * 0.47 + learned * 0.33 + sample * 0.20
                if best is None or score > best[0]:
                    best = (score, units, option, level)
        if best:
            result = (best[1], best[0], best[2], best[3])
            self._best_final_cache[cache_key] = result
            return result
        centroid = self._known_units(list(cluster.get("centroidUnits") or []))[:8]
        result = (centroid, self._heuristic_board_score(centroid, set(), False), None, len(centroid) or 8)
        self._best_final_cache[cache_key] = result
        return result

    def _observed_level_boards(self, cluster: dict[str, Any], level: int) -> list[dict[str, Any]]:
        """Merge early/final observed boards for one level and keep only strong distinct nodes."""
        cache_key = (str(cluster.get("id") or id(cluster)), int(level))
        cached = self._level_board_cache.get(cache_key)
        if cached is not None:
            return cached
        output: dict[tuple[str, ...], dict[str, Any]] = {}
        for source_key in ("earlyOptions", "options"):
            for option in (cluster.get(source_key) or {}).get(str(level)) or []:
                units = self._known_units(list(option.get("units") or []))
                if len(units) < 3:
                    continue
                key = tuple(sorted(units))
                observed = placement_strength(float(option.get("avg") or 0))
                learned = float(option.get("ml") if option.get("ml") is not None else self._heuristic_board_score(units, set(), level <= 6))
                sample = evidence(int(option.get("count") or 0))
                size_fit = clamp(1.0 - abs(len(units) - level) * 0.14)
                node_score = observed * 0.37 + learned * 0.31 + sample * 0.22 + size_fit * 0.10
                row = {
                    "level": level,
                    "units": units,
                    "score": node_score,
                    "avgPlacement": float(option.get("avg") or 0) or None,
                    "games": int(option.get("count") or 0),
                }
                previous = output.get(key)
                if previous is None or node_score > float(previous["score"]):
                    output[key] = row
        result = sorted(output.values(), key=lambda value: float(value["score"]), reverse=True)[:24]
        self._level_board_cache[cache_key] = result
        return result

    def _transition_path(
        self,
        cluster: dict[str, Any],
        current_level: int,
        owned: set[str],
        early_board: list[str],
        final_board: list[str],
        final_level: int,
    ) -> tuple[list[dict[str, Any]], float]:
        """Dynamic-programming path that rewards strength while minimizing unnecessary swaps."""
        start = {
            "level": current_level,
            "units": early_board,
            "score": self._heuristic_board_score(early_board, owned, True),
            "avgPlacement": None,
            "games": 0,
        }
        all_levels = sorted({
            int(key)
            for source_key in ("earlyOptions", "options")
            for key in (cluster.get(source_key) or {})
            if str(key).isdigit() and current_level < int(key) <= max(current_level, final_level)
        })
        if final_level > current_level and final_level not in all_levels:
            all_levels.append(final_level)
            all_levels.sort()
        states: list[tuple[float, list[dict[str, Any]]]] = [(float(start["score"]), [start])]
        for level in all_levels:
            nodes = self._observed_level_boards(cluster, level)
            if level == final_level and final_board:
                final_key = tuple(sorted(final_board))
                if all(tuple(sorted(node["units"])) != final_key for node in nodes):
                    nodes.append({
                        "level": level,
                        "units": final_board,
                        "score": self._heuristic_board_score(final_board, set(), False),
                        "avgPlacement": None,
                        "games": 0,
                    })
            if not nodes:
                continue
            next_states: list[tuple[float, list[dict[str, Any]]]] = []
            for total, path in states:
                previous = set(path[-1]["units"])
                for node in nodes:
                    current = set(node["units"])
                    retained = len(previous & current) / max(1, min(len(previous), len(current)))
                    swap_rate = 1.0 - len(previous & current) / max(1, len(previous | current))
                    transition = retained * 0.16 - swap_rate * 0.11
                    next_states.append((total + float(node["score"]) + transition, path + [node]))
            next_states.sort(key=lambda value: value[0], reverse=True)
            states = next_states[:64]
        if not states:
            return [start], 0.5
        total, path = max(states, key=lambda value: value[0])
        normalized = clamp(total / max(1, len(path)))
        return path, normalized

    def _fallback_position_distribution(self, unit_id: str) -> list[float]:
        champion = self.catalog.champion_by_id.get(unit_id, {})
        role = str(champion.get("role") or "")
        stats = champion.get("stats") or {}
        attack_range = float(stats.get("range") or stats.get("attackRange") or 1)
        values = [0.005] * CELL_COUNT
        if "Tank" in role or "Fighter" in role:
            row_weights = [0.04, 0.10, 0.28, 0.58]
        elif "Reaper" in role:
            row_weights = [0.08, 0.30, 0.43, 0.19]
        elif role.startswith("AD") or role.startswith("AP") or attack_range >= 3:
            row_weights = [0.62, 0.26, 0.09, 0.03]
        else:
            row_weights = [0.30, 0.30, 0.24, 0.16]
        for row in range(4):
            for col in range(7):
                # A tiny center preference keeps fallback boards compact without pretending
                # we know the opponent's side. Live/model distributions override this.
                center = 1.0 - abs(col - 3) * 0.035
                values[row * 7 + col] += row_weights[row] * center / 7.0
        total = sum(values)
        return [value / total for value in values]

    def _empirical_position_distribution(self, cluster: dict[str, Any], unit_id: str) -> tuple[list[float], int]:
        values = [0.0] * CELL_COUNT
        positions = ((cluster.get("positioning") or {}).get("units") or {}).get(unit_id) or []
        for row in positions:
            cell = str(row.get("cell") or "")
            if not cell.startswith("cell_"):
                continue
            try:
                index = int(cell.split("_", 1)[1]) - 1
            except (IndexError, ValueError):
                continue
            if 0 <= index < CELL_COUNT:
                values[index] += max(0, int(row.get("count") or 0))
        count = int(sum(values))
        if count > 0:
            values = [value / count for value in values]
        return values, count

    def _position_board(self, cluster: dict[str, Any], board_units: list[str]) -> list[dict[str, Any]]:
        board = self._known_units(board_units)
        if not board:
            return []
        empirical_by_unit = {
            unit_id: self._empirical_position_distribution(cluster, unit_id)
            for unit_id in board
        }
        # Exact live cluster positioning is stronger evidence than a generalized model and is
        # also effectively free at request time. Invoke TensorFlow only for sparse/missing
        # unit-cluster distributions where generalization actually adds value.
        needs_model = [
            unit_id for unit_id in board
            if empirical_by_unit[unit_id][1] < 80
        ]
        model_results = self.position_ranker.predict_many([(board, unit_id) for unit_id in needs_model])
        prediction_by_unit = dict(zip(needs_model, model_results, strict=True))
        distributions: dict[str, list[float]] = {}
        evidence_counts: dict[str, int] = {}
        model_stds: dict[str, float] = {}
        sources: dict[str, str] = {}

        for unit_id in board:
            empirical, count = empirical_by_unit[unit_id]
            prediction = prediction_by_unit.get(unit_id)
            evidence_counts[unit_id] = count
            model = list(prediction[0]) if prediction is not None else []
            model_std = float(sum(prediction[1]) / len(prediction[1])) if prediction is not None else 0.0
            model_stds[unit_id] = model_std
            if count >= 80:
                values = empirical
                sources[unit_id] = "live"
            elif count > 0 and model:
                empirical_weight = 0.66 + evidence(count) * 0.22
                values = [
                    empirical[index] * empirical_weight + float(model[index]) * (1.0 - empirical_weight)
                    for index in range(CELL_COUNT)
                ]
                sources[unit_id] = "live+ml"
            elif count > 0:
                values = empirical
                sources[unit_id] = "live"
            elif model:
                values = [float(value) for value in model]
                sources[unit_id] = "ml"
            else:
                values = self._fallback_position_distribution(unit_id)
                sources[unit_id] = "role-fallback"
            total = sum(values)
            distributions[unit_id] = [value / total for value in values] if total > 0 else self._fallback_position_distribution(unit_id)

        # Most constrained units are assigned first. Beam search maximizes the joint
        # likelihood while enforcing one champion per hex.
        ordered = sorted(board, key=lambda unit_id: max(distributions[unit_id]), reverse=True)
        beams: list[tuple[float, dict[str, int], frozenset[int]]] = [(0.0, {}, frozenset())]
        for unit_id in ordered:
            distribution = distributions[unit_id]
            ranked_cells = sorted(range(CELL_COUNT), key=lambda index: distribution[index], reverse=True)
            candidate_cells = ranked_cells[:10]
            next_beams: list[tuple[float, dict[str, int], frozenset[int]]] = []
            for score, assignment, used in beams:
                available = [cell for cell in candidate_cells if cell not in used]
                if not available:
                    available = [cell for cell in ranked_cells if cell not in used][:4]
                for cell in available:
                    probability = max(1e-7, distribution[cell])
                    next_assignment = dict(assignment)
                    next_assignment[unit_id] = cell
                    next_beams.append((score + math.log(probability), next_assignment, used | {cell}))
            next_beams.sort(key=lambda value: value[0], reverse=True)
            beams = next_beams[:512]
        assignment = beams[0][1] if beams else {}

        output: list[dict[str, Any]] = []
        for unit_id in board:
            cell = assignment.get(unit_id)
            if cell is None:
                continue
            distribution = distributions[unit_id]
            peak = max(distribution)
            assigned = distribution[cell]
            source_row = cell // 7
            row, col = display_grid_position(cell)
            row_probability = sum(distribution[source_row * 7 : source_row * 7 + 7])
            count = evidence_counts[unit_id]
            evidence_factor = 0.62 + evidence(count) * 0.38 if count else 0.62
            model_factor = clamp(1.0 - model_stds[unit_id] / 0.035, 0.72, 1.0)
            confidence = clamp((assigned / max(1e-8, peak)) * row_probability * evidence_factor * model_factor)
            alternatives = sorted(range(CELL_COUNT), key=lambda index: distribution[index], reverse=True)[:4]
            output.append({
                "unitId": unit_id,
                "cell": cell_name(cell),
                "row": row,
                "col": col,
                "confidence": round(confidence * 100, 1),
                "source": sources[unit_id],
                "sampleCount": count,
                "alternatives": [
                    {"cell": cell_name(index), "probability": round(distribution[index] * 100, 1)}
                    for index in alternatives
                    if index != cell
                ][:3],
            })
        return output

    @staticmethod
    def _role_is_tank(role: str) -> bool:
        return "Tank" in role or "Fighter" in role

    @staticmethod
    def _role_is_ad(role: str) -> bool:
        return role.startswith("AD") and "Tank" not in role

    @staticmethod
    def _role_is_ap(role: str) -> bool:
        return role.startswith("AP") and "Tank" not in role

    def _emblem_trait_id(self, item: dict[str, Any]) -> str | None:
        item_id = str(item.get("id") or "")
        if item_id in self._emblem_trait_cache:
            return self._emblem_trait_cache[item_id]
        if item.get("category") != "emblem":
            self._emblem_trait_cache[item_id] = None
            return None
        names = {str(value).strip().lower() for value in item.get("traits") or [] if value}
        name_en = str(item.get("nameEn") or "").removesuffix(" Emblem").strip().lower()
        if name_en:
            names.add(name_en)
        for trait in self.catalog.traits:
            if str(trait.get("nameEn") or "").strip().lower() in names:
                trait_id = str(trait["id"])
                self._emblem_trait_cache[item_id] = trait_id
                return trait_id
        self._emblem_trait_cache[item_id] = None
        return None

    def _emblem_holder_score(self, item: dict[str, Any], holder_id: str, board: list[str]) -> float:
        trait_id = self._emblem_trait_id(item)
        holder = self.catalog.champion_by_id.get(holder_id)
        trait = self.catalog.trait_by_id.get(trait_id or "")
        if not trait_id or not holder or not trait or trait_id in (holder.get("traits") or []):
            return 0.0
        current = sum(
            trait_id in (self.catalog.champion_by_id.get(unit_id, {}).get("traits") or [])
            for unit_id in board
        )
        breakpoints = sorted(int(value) for value in trait.get("breakpoints") or [])
        before = max((value for value in breakpoints if value <= current), default=0)
        after = max((value for value in breakpoints if value <= current + 1), default=0)
        next_before = next((value for value in breakpoints if value > current), None)
        score = 0.32
        if after > before:
            score += 0.54
        elif next_before is not None:
            gap_before = max(1, next_before - current)
            gap_after = max(0, next_before - (current + 1))
            score += 0.24 * (gap_before - gap_after) / gap_before
        # Prefer carrying an emblem on a unit that is not one of the primary 3-item damage
        # carries when the trait gain is identical, preserving item slots on the real carry.
        role = str(holder.get("role") or "")
        if "Tank" in role or "Support" in role:
            score += 0.06
        return clamp(score)

    def _item_archetype(self, item: dict[str, Any]) -> str:
        if item.get("category") == "emblem":
            return "emblem"
        name = str(item.get("nameEn") or "").lower()
        if any(token in name for token in ("infinity edge", "deathblade", "last whisper", "guinsoo", "kraken", "red buff")):
            return "ad"
        if any(token in name for token in ("shojin", "blue buff", "archangel", "jeweled gauntlet", "rabadon", "nashor")):
            return "ap"
        if any(token in name for token in ("warmog", "gargoyle", "bramble", "dragon's claw", "sunfire", "evenshroud", "steadfast", "spirit visage")):
            return "tank"
        return "flex"

    def _strict_role_mismatch(self, item: dict[str, Any], holder_id: str) -> bool:
        holder = self.catalog.champion_by_id.get(holder_id)
        if not holder:
            return True
        role = str(holder.get("role") or "")
        archetype = self._item_archetype(item)
        if archetype == "ad":
            return not self._role_is_ad(role)
        if archetype == "ap":
            return not self._role_is_ap(role)
        if archetype == "tank":
            return not self._role_is_tank(role)
        return False

    def _item_role_score(self, item: dict[str, Any], holder_id: str) -> float:
        cache_key = (str(item.get("id") or ""), holder_id)
        cached = self._item_role_cache.get(cache_key)
        if cached is not None:
            return cached
        holder = self.catalog.champion_by_id.get(holder_id)
        if not holder:
            return 0.0
        tags = {str(value).lower() for value in item.get("tags") or []}
        effect_keys = {str(value).lower() for value in (item.get("effects") or {})}
        role = str(holder.get("role") or "")
        score = 0.18
        tank_signal = any(
            token in key for key in effect_keys
            for token in ("armor", "magicresist", "health", "damage reduction", "damagereduction")
        ) or any(token in tags for token in ("health", "armor", "magicresist", "tank"))
        ad_signal = any(
            token in key for key in effect_keys
            for token in ("attackdamage", "attackspeed", "critchance", "criticalstrike")
        ) or any(token in tags for token in ("attackdamage", "attackspeed", "critchance", "attack"))
        ap_signal = any(
            key in {"ap", "abilitypower", "mana", "manaperattack", "manapersecond", "manaregen", "manaregenbase"}
            or "abilitypower" in key
            for key in effect_keys
        ) or any(token in tags for token in ("abilitypower", "mana", "magic"))
        if tank_signal and self._role_is_tank(role):
            score += 0.42
        if ad_signal and self._role_is_ad(role):
            score += 0.38
        if ap_signal and self._role_is_ap(role):
            score += 0.38
        if "stacking" in tags and any(token in role for token in ("Carry", "Caster")):
            score += 0.10
        name = str(item.get("nameEn") or "").lower()
        if any(token in name for token in ("guinsoo", "kraken", "red buff", "last whisper", "infinity edge", "deathblade")):
            score += 0.24 if self._role_is_ad(role) else -0.12
        if any(token in name for token in ("shojin", "blue buff", "archangel", "jeweled gauntlet", "rabadon")):
            score += 0.24 if self._role_is_ap(role) else -0.08
        if any(token in name for token in ("warmog", "gargoyle", "bramble", "dragon's claw", "sunfire", "evenshroud", "steadfast")):
            score += 0.24 if self._role_is_tank(role) else -0.10
        result = clamp(score)
        self._item_role_cache[cache_key] = result
        return result

    @staticmethod
    def _pair_affinity(pair_count: int, left_support: int, right_support: int) -> float:
        if pair_count <= 0 or left_support <= 0 or right_support <= 0:
            return 0.0
        cosine = pair_count / max(1.0, math.sqrt(left_support * right_support))
        conditional = pair_count / max(1.0, float(min(left_support, right_support)))
        return clamp(cosine * 0.60 + conditional * 0.40)

    def _item_unique_effects(self, item_id: str) -> set[str]:
        """Infer important non-stacking team effects from Riot item effect keys.

        This intentionally avoids name-based special cases. Current Set 18 data exposes the
        examples we care about as Wound (Sunfire/Red Buff/Morello) and Sunder
        (Evenshroud/Last Whisper). Shred is included for the analogous MR-reduction family.
        """
        item = self.catalog.item_by_id.get(item_id, {})
        keys = {str(key).lower() for key in (item.get("effects") or {})}
        groups: set[str] = set()
        if any("wound" in key for key in keys):
            groups.add("wound")
        if any("sunder" in key for key in keys):
            groups.add("sunder")
        if any("shred" in key for key in keys):
            groups.add("shred")
        return groups

    def _item_pair_empirical(
        self,
        holder_id: str | None,
        left_item_id: str,
        right_item_id: str,
    ) -> tuple[float, int, int, bool]:
        """Return affinity, individual support, pair count, and holder-specific flag."""
        left, right = sorted((left_item_id, right_item_id))
        pair = (left, right)
        if holder_id:
            holder_items = self._pair_holder_items.get(holder_id)
            holder_pairs = self._pair_holder_pairs.get(holder_id)
            if holder_items is not None and holder_pairs is not None:
                support = min(int(holder_items[left]), int(holder_items[right]))
                if support >= 12:
                    pair_count = int(holder_pairs[pair])
                    return (
                        self._pair_affinity(pair_count, int(holder_items[left]), int(holder_items[right])),
                        support,
                        pair_count,
                        True,
                    )
        support = min(int(self._pair_global_items[left]), int(self._pair_global_items[right]))
        pair_count = int(self._pair_global_pairs[pair])
        return (
            self._pair_affinity(
                pair_count,
                int(self._pair_global_items[left]),
                int(self._pair_global_items[right]),
            ),
            support,
            pair_count,
            False,
        )

    def _item_pair_compatibility(
        self,
        holder_id: str | None,
        left_item_id: str,
        right_item_id: str,
    ) -> tuple[float, int, int, set[str], bool]:
        if left_item_id == right_item_id:
            return 0.0, 0, 0, set(), False
        if left_item_id not in self.catalog.item_by_id or right_item_id not in self.catalog.item_by_id:
            # Keep synthetic/test candidates and future unknown payloads neutral rather than
            # asking the learned model to extrapolate outside its item vocabulary.
            return 0.5, 0, 0, set(), False
        left, right = sorted((left_item_id, right_item_id))
        empirical, support, pair_count, holder_specific = self._item_pair_empirical(holder_id, left, right)
        cache_key = (holder_id or "*", left, right)
        learned = self._item_pair_score_cache.get(cache_key)

        if support >= 80:
            score = empirical
        elif support > 0 and learned is not None:
            empirical_weight = 0.48 + evidence(support) * 0.34
            score = empirical * empirical_weight + learned * (1.0 - empirical_weight)
        elif support > 0:
            score = empirical
        elif learned is not None:
            score = learned
        else:
            score = 0.5

        overlap = self._item_unique_effects(left) & self._item_unique_effects(right)
        if overlap:
            # A non-stacking effect is a safety prior, not an unconditional blacklist. Strong
            # observed co-occurrence can override most of the prior on a future patch.
            observed_pair_evidence = evidence(pair_count)
            prior_weight = 0.52 * (1.0 - 0.72 * observed_pair_evidence)
            score = score * (1.0 - prior_weight) + 0.06 * prior_weight
        return clamp(score), support, pair_count, overlap, holder_specific

    def _items_can_coexist(
        self,
        holder_id: str | None,
        left_item_id: str,
        right_item_id: str,
    ) -> bool:
        compatibility, support, pair_count, overlap, holder_specific = self._item_pair_compatibility(
            holder_id,
            left_item_id,
            right_item_id,
        )
        if overlap:
            # Team-wide duplicate Wound/Sunder/Shred is normally wasted. Keep an escape hatch
            # for patches where live builds genuinely pair them at meaningful volume.
            return pair_count >= 20 and compatibility >= 0.55
        # For ordinary item pairs, only reject a zero-cooccurrence pair when the *same holder*
        # has strong evidence for both items independently. Sparse/global data remains soft.
        if holder_specific and support >= 35 and pair_count == 0 and compatibility < 0.12:
            return False
        return True

    @staticmethod
    def _can_craft(recipe: list[str], components: list[str]) -> bool:
        available = Counter(components)
        for component in recipe:
            if available[component] <= 0:
                return False
            available[component] -= 1
        return True

    def _item_stat(self, cluster: dict[str, Any], item_id: str, holder_id: str | None = None) -> tuple[float, int, float]:
        cache_key = (str(cluster.get("id") or id(cluster)), item_id, holder_id or "*")
        cached = self._item_stat_cache.get(cache_key)
        if cached is not None:
            return cached
        for row in cluster.get("itemStats") or []:
            if row.get("item") != item_id:
                continue
            if holder_id:
                for holder in row.get("units") or []:
                    if holder.get("unit") == holder_id:
                        avg = float(holder.get("avg") or 0)
                        result = placement_strength(avg), int(holder.get("count") or 0), float(holder.get("placeChange") or 0)
                        self._item_stat_cache[cache_key] = result
                        return result
                # Global item strength is not evidence that an arbitrary champion is a good
                # holder. Falling through here used to create false-positive holder matches.
                result = (0.5, 0, 0.0)
                self._item_stat_cache[cache_key] = result
                return result
            result = placement_strength(float(row.get("avg") or 0)), int(row.get("count") or 0), 0.0
            self._item_stat_cache[cache_key] = result
            return result
        result = (0.5, 0, 0.0)
        self._item_stat_cache[cache_key] = result
        return result

    def _craftable_completed_items(self, components: list[str]) -> list[dict[str, Any]]:
        if len(components) < 2:
            return []
        return [
            item for item in self.catalog.items
            if item.get("category") in {"completed", "emblem"}
            and len(item.get("composition") or []) == 2
            and self._can_craft(list(item.get("composition") or []), components)
        ]

    def _rank_holders(self, cluster: dict[str, Any], item: dict[str, Any], candidates: list[str]) -> list[tuple[str, float, str]]:
        ranked: list[tuple[str, float, str]] = []
        for holder_id in unique(candidates):
            if holder_id not in self.catalog.champion_by_id:
                continue
            emblem_score = self._emblem_holder_score(item, holder_id, candidates)
            if item.get("category") == "emblem" and emblem_score <= 0:
                continue
            empirical, count, place_change = self._item_stat(cluster, str(item["id"]), holder_id)
            role = self._item_role_score(item, holder_id)
            sample = evidence(count)
            improvement = clamp((-place_change + 1.0) / 2.0) if count else 0.5
            opgg, opgg_ev = self._opgg_item_strength(str(item["id"]), holder_id)
            pro, pro_ev = self._high_elo_item_holder_prior(str(item["id"]), holder_id)
            affinity_key = (holder_id, str(item["id"]))
            affinity = self._item_affinity_scores.get(affinity_key)
            if affinity is None:
                affinity = self.item_affinity_ranker.score_affinity(*affinity_key)
            signals: list[tuple[float, float]] = [
                (role, 0.28),
                (empirical, 0.24),
                (sample, 0.12),
                (improvement, 0.12),
            ]
            if affinity is not None:
                signals.append((affinity, 0.24))
            if item.get("category") == "emblem":
                signals.append((emblem_score, 0.55))
            if opgg_ev > 0:
                signals.extend([(opgg, 0.12), (opgg_ev, 0.04)])
            if pro_ev > 0:
                signals.extend([(pro, 0.06), (pro_ev, 0.02)])
            denominator = sum(weight for _, weight in signals)
            score = sum(value * weight for value, weight in signals) / denominator
            # Do not let weak global statistics put pure carry items on tanks/supports. Real
            # holder evidence can still override this for an unusual patch/augment case.
            if self._strict_role_mismatch(item, holder_id):
                strong_holder_evidence = count >= 30 and improvement >= 0.58
                strong_affinity = affinity is not None and affinity >= 0.67
                if not (strong_holder_evidence and strong_affinity):
                    score *= 0.34
            is_emblem = item.get("category") == "emblem"
            reason = "hợp vai trò"
            if is_emblem:
                trait_id = self._emblem_trait_id(item)
                trait = self.catalog.trait_by_id.get(trait_id or "", {})
                reason = f"Ấn {trait.get('name') or item.get('name')} tăng mốc tộc/hệ"
            elif affinity is not None and affinity >= 0.66 and count >= 5:
                reason = f"holder affinity {int(affinity * 100)}% · {count} mẫu live"
            if not is_emblem and count >= 5 and opgg_ev > 0 and pro_ev > 0:
                reason = f"MetaTFT {count} mẫu + OP.GG + high-Elo xác nhận"
            elif not is_emblem and count >= 5 and opgg_ev > 0:
                reason = f"MetaTFT {count} mẫu + OP.GG xác nhận holder"
            elif not is_emblem and count >= 5 and pro_ev > 0:
                reason = f"MetaTFT {count} mẫu + high-Elo đang dùng"
            elif not is_emblem and count >= 5:
                reason = f"{count} mẫu holder live, avg {8.5 - empirical * 7.5:.2f}"
            ranked.append((holder_id, clamp(score), reason))
        ranked.sort(key=lambda value: value[1], reverse=True)
        return ranked

    def _best_holder(self, cluster: dict[str, Any], item: dict[str, Any], candidates: list[str]) -> tuple[str | None, float, str]:
        ranked = self._rank_holders(cluster, item, candidates)
        return ranked[0] if ranked else (None, 0.0, "")

    def _item_core_priority(self, cluster: dict[str, Any], item_id: str, holder_id: str | None) -> float:
        """Estimate how dangerous it is to consume this item's components on a weaker slam.

        A core item is one that repeatedly appears in strong observed 3-item builds for the
        intended late holder. This is intentionally holder-specific: a Sword may be expendable
        in one comp but critical when it is the only route to the main carry's IE/Deathblade.
        """
        if not holder_id:
            return 0.0
        holder_builds = [
            row for row in cluster.get("builds") or []
            if str(row.get("unit") or "") == holder_id and row.get("items")
        ]
        if not holder_builds:
            affinity = self._item_affinity_scores.get((holder_id, item_id))
            return clamp((affinity or 0.0) * 0.55)

        total_support = sum(max(0, int(row.get("count") or 0)) for row in holder_builds)
        matching = [row for row in holder_builds if item_id in (row.get("items") or [])]
        if not matching:
            affinity = self._item_affinity_scores.get((holder_id, item_id))
            return clamp((affinity or 0.0) * 0.45)

        item_support = sum(max(0, int(row.get("count") or 0)) for row in matching)
        prevalence = item_support / max(1, total_support)
        best_quality = 0.0
        for row in matching:
            observed = placement_strength(float(row.get("avg") or 0))
            learned = float(row.get("ml")) if row.get("ml") is not None else observed
            count = int(row.get("count") or 0)
            quality = observed * 0.50 + learned * 0.24 + evidence(count) * 0.26
            best_quality = max(best_quality, quality)
        affinity = self._item_affinity_scores.get((holder_id, item_id))
        if affinity is None:
            affinity = self.item_affinity_ranker.score_affinity(holder_id, item_id)
        return clamp(
            best_quality * 0.38
            + evidence(item_support) * 0.27
            + clamp(prevalence * 2.4) * 0.25
            + (affinity or 0.5) * 0.10
        )

    def _select_craft_set(
        self,
        ranked: list[dict[str, Any]],
        components: list[str],
        limit: int = 3,
        cluster: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Small exact search: maximize recommendation score without reusing components."""
        if not ranked or len(components) < 2:
            return []
        inventory = Counter(components)
        candidates = ranked[:14]
        best_score = -1.0
        best_rows: list[dict[str, Any]] = []

        def dfs(index: int, left: Counter[str], chosen: list[dict[str, Any]], total: float) -> None:
            nonlocal best_score, best_rows
            if total > best_score:
                best_score = total
                best_rows = list(chosen)
            if index >= len(candidates) or len(chosen) >= limit:
                return
            for candidate_index in range(index, len(candidates)):
                row = candidates[candidate_index]
                item_id = str(row.get("itemId") or "")
                holder_id = str(row.get("holderId") or "")
                if cluster is not None and item_id:
                    incompatible = False
                    for previous in chosen:
                        previous_item = str(previous.get("itemId") or "")
                        if not previous_item:
                            continue
                        previous_holder = str(previous.get("holderId") or "")
                        pair_holder = holder_id if holder_id and holder_id == previous_holder else None
                        if not self._items_can_coexist(pair_holder, item_id, previous_item):
                            incompatible = True
                            break
                    if incompatible:
                        continue
                recipe = list(row.get("recipe") or [])
                need = Counter(recipe)
                if any(left[component] < amount for component, amount in need.items()):
                    continue
                next_left = left.copy()
                next_left.subtract(need)
                dfs(candidate_index + 1, next_left, chosen + [row], total + float(row["score"]))

        dfs(0, inventory, [], 0.0)
        return best_rows

    def _select_stage_item_set(
        self,
        ranked: list[dict[str, Any]],
        components: list[str],
        limit: int = 3,
        cluster: dict[str, Any] | None = None,
        stage_id: str = "late",
        preserve_stability: bool = True,
        previous_rows: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Jointly choose recipes *and* holders, rewarding coherent 2/3-item bundles.

        Independent per-item holder selection tends to spread three useful carry items across
        three champions. In actual TFT, concentrating a coherent 2/3-item build on the main
        carry or main tank is usually stronger. The search still respects component inventory
        exactly and caps each holder at three items.
        """
        if not ranked or len(components) < 2:
            return []
        inventory = Counter(components)
        candidates = list(ranked[:48])
        # Never prune an item that was present in the immediately previous backend plan.
        # Adding one component can create many new item/holder alternatives and used to push
        # the old recommendation beyond ranked[:36], making it impossible for hysteresis to
        # preserve it even though the recipe was still legal.
        candidate_keys = {
            (str(row.get("itemId") or ""), str(row.get("holderId") or ""))
            for row in candidates
        }
        for previous in previous_rows or []:
            item_id = str(previous.get("itemId") or "")
            holder_id = str(previous.get("holderId") or "")
            matches = [row for row in ranked if str(row.get("itemId") or "") == item_id]
            if not matches:
                continue
            preferred = next(
                (row for row in matches if str(row.get("holderId") or "") == holder_id),
                matches[0],
            )
            key = (str(preferred.get("itemId") or ""), str(preferred.get("holderId") or ""))
            if key not in candidate_keys:
                candidates.append(preferred)
                candidate_keys.add(key)
        best_score = 0.0
        best_rows: list[dict[str, Any]] = []
        hold_threshold = {"opener": 70.0, "mid": 66.0, "late": 60.0}.get(stage_id, 60.0)

        # Recommendation stability: adding an unrelated component must not make an already
        # valid slam disappear simply because the global optimum changed by a tiny amount.
        # Derive anchors from one-component-smaller neighboring inventories. We only anchor
        # rows whose recipe does NOT use the removed component, so a genuinely new recipe is
        # still free to replace an old one when it is materially better.
        stable_anchor_counts: Counter[tuple[str, str]] = Counter()
        stable_item_counts: Counter[str] = Counter()
        stable_anchor_rows: dict[str, dict[str, Any]] = {}
        history_item_ids: set[str] = set()
        if preserve_stability and len(components) >= 3:
            seen_reduced: set[tuple[str, ...]] = set()
            for remove_index, removed_component in enumerate(components):
                reduced_components = components[:remove_index] + components[remove_index + 1 :]
                reduced_key = tuple(sorted(reduced_components))
                if reduced_key in seen_reduced or len(reduced_components) < 2:
                    continue
                seen_reduced.add(reduced_key)
                reduced_inventory = Counter(reduced_components)
                reduced_ranked = [
                    row for row in candidates
                    if all(
                        reduced_inventory[component] >= amount
                        for component, amount in Counter(row.get("recipe") or []).items()
                    )
                ]
                if not reduced_ranked:
                    continue
                reduced_selected = self._select_stage_item_set(
                    reduced_ranked,
                    reduced_components,
                    limit=limit,
                    cluster=cluster,
                    stage_id=stage_id,
                    preserve_stability=False,
                )
                for row in reduced_selected:
                    recipe = list(row.get("recipe") or [])
                    if removed_component in recipe:
                        continue
                    item_id = str(row.get("itemId") or "")
                    holder_id = str(row.get("holderId") or "")
                    if not item_id:
                        continue
                    stable_item_counts[item_id] += 1
                    previous_anchor = stable_anchor_rows.get(item_id)
                    if previous_anchor is None or (
                        float(row.get("score") or 0.0) + float(row.get("corePriority") or 0.0) * 20.0
                    ) > (
                        float(previous_anchor.get("score") or 0.0)
                        + float(previous_anchor.get("corePriority") or 0.0) * 20.0
                    ):
                        stable_anchor_rows[item_id] = row
                    if holder_id:
                        stable_anchor_counts[(item_id, holder_id)] += 1

        # Exact interaction history is stronger than inferred neighboring inventories. The
        # frontend only echoes the backend's previous output; all preserve/replace decisions
        # remain here. If the old item is still legal in the enlarged component bag, give it a
        # strong hysteresis anchor so a side-grade cannot make it flicker away.
        for previous in previous_rows or []:
            item_id = str(previous.get("itemId") or "")
            if not item_id:
                continue
            same_item = [row for row in candidates if str(row.get("itemId") or "") == item_id]
            if not same_item:
                continue
            previous_holder = str(previous.get("holderId") or "")
            anchor = next(
                (row for row in same_item if str(row.get("holderId") or "") == previous_holder),
                same_item[0],
            )
            stable_anchor_rows[item_id] = anchor
            stable_item_counts[item_id] += 5
            history_item_ids.add(item_id)
            holder_id = str(anchor.get("holderId") or "")
            if holder_id:
                stable_anchor_counts[(item_id, holder_id)] += 5

        history_seed: list[dict[str, Any]] = []
        history_left = inventory.copy()
        history_holder_counts: Counter[str] = Counter()
        if previous_rows:
            valid_history = True
            for previous in previous_rows:
                item_id = str(previous.get("itemId") or "")
                previous_holder = str(previous.get("holderId") or "")
                matches = [row for row in candidates if str(row.get("itemId") or "") == item_id]
                if not matches:
                    valid_history = False
                    break
                row = next(
                    (value for value in matches if str(value.get("holderId") or "") == previous_holder),
                    matches[0],
                )
                holder_id = str(row.get("holderId") or "")
                if not holder_id or history_holder_counts[holder_id] >= 3:
                    valid_history = False
                    break
                need = Counter(row.get("recipe") or [])
                if any(history_left[component] < amount for component, amount in need.items()):
                    valid_history = False
                    break
                incompatible = False
                for selected in history_seed:
                    selected_item = str(selected.get("itemId") or "")
                    selected_holder = str(selected.get("holderId") or "")
                    pair_holder = holder_id if holder_id == selected_holder else None
                    if selected_item and not self._items_can_coexist(pair_holder, item_id, selected_item):
                        incompatible = True
                        break
                if incompatible:
                    valid_history = False
                    break
                history_left.subtract(need)
                history_holder_counts[holder_id] += 1
                history_seed.append(row)
            if not valid_history:
                history_seed = []
                history_left = inventory.copy()
                history_holder_counts = Counter()

        # Keep one representative for each high-value core craft. These rows are used only to
        # price the opportunity cost of consuming their components on lower-value items.
        protected_by_item: dict[str, dict[str, Any]] = {}
        for row in candidates:
            item_id = str(row.get("itemId") or "")
            core = float(row.get("corePriority") or 0.0)
            if not item_id or core < 0.68:
                continue
            previous = protected_by_item.get(item_id)
            if previous is None or (core, float(row.get("score") or 0.0)) > (
                float(previous.get("corePriority") or 0.0),
                float(previous.get("score") or 0.0),
            ):
                protected_by_item[item_id] = row
        protected = list(protected_by_item.values())

        def opportunity_penalty(left: Counter[str], chosen: list[dict[str, Any]]) -> float:
            if stage_id == "late" or not protected:
                return 0.0
            selected_ids = {str(row.get("itemId") or "") for row in chosen}
            penalty = 0.0
            for core_row in protected:
                core_item = str(core_row.get("itemId") or "")
                if core_item in selected_ids:
                    continue
                recipe = list(core_row.get("recipe") or [])
                if not recipe:
                    continue
                # Only charge regret if this core item was craftable from the original bag but
                # has become impossible because an already-selected slam consumed a component.
                original_need = Counter(recipe)
                if any(inventory[component] < amount for component, amount in original_need.items()):
                    continue
                if all(left[component] >= amount for component, amount in original_need.items()):
                    continue
                core_priority = float(core_row.get("corePriority") or 0.0)
                core_score = float(core_row.get("score") or 0.0)
                # A similarly important selected core item is a legitimate alternative use of
                # the contested component, so reduce the regret instead of protecting both.
                replacement = 0.0
                core_recipe = set(recipe)
                for selected in chosen:
                    selected_recipe = set(selected.get("recipe") or [])
                    if not core_recipe.intersection(selected_recipe):
                        continue
                    replacement = max(replacement, float(selected.get("corePriority") or 0.0))
                regret_gap = max(0.0, core_priority - replacement * 0.88)
                penalty += regret_gap * 30.0 + max(0.0, core_score - hold_threshold) * regret_gap * 0.28
            return penalty

        def stability_penalty(left: Counter[str], chosen: list[dict[str, Any]]) -> float:
            if not stable_anchor_rows:
                return 0.0
            selected_ids = {str(row.get("itemId") or "") for row in chosen}
            penalty = 0.0
            for item_id, anchor in stable_anchor_rows.items():
                if item_id in selected_ids:
                    continue
                recipe = list(anchor.get("recipe") or [])
                if not recipe:
                    continue
                anchor_score = float(anchor.get("score") or 0.0)
                anchor_core = float(anchor.get("corePriority") or 0.0)
                anchor_value = anchor_score + anchor_core * 20.0
                need = Counter(recipe)

                # Find what actually displaced this stable craft. If it is merely a side-grade,
                # keep the old recommendation. A replacement needs roughly +10 effective
                # points (score + core value) before churn is allowed.
                replacement_value = -1.0
                anchor_recipe = set(recipe)
                for selected in chosen:
                    selected_recipe = set(selected.get("recipe") or [])
                    if not anchor_recipe.intersection(selected_recipe):
                        continue
                    value = float(selected.get("score") or 0.0) + float(selected.get("corePriority") or 0.0) * 20.0
                    replacement_value = max(replacement_value, value)

                if item_id in history_item_ids:
                    # This item was literally on the user's previous backend result. When the
                    # component bag only grew, preserve it unless the conflicting replacement
                    # is a decisive upgrade rather than a side-grade. This prevents visible
                    # item flicker while still allowing a clearly superior newly-unlocked BIS.
                    if replacement_value >= anchor_value + 15.0:
                        continue
                    improvement = max(0.0, replacement_value - anchor_value)
                    penalty += max(32.0, 64.0 - improvement * 1.5)
                    continue
                if replacement_value >= anchor_value + 10.0:
                    continue
                if all(left[component] >= amount for component, amount in need.items()) and len(chosen) < limit:
                    # It still fits and there is a free recommendation slot: omitting it is
                    # pure churn, not a trade-off.
                    penalty += 22.0
                else:
                    improvement = max(0.0, replacement_value - anchor_value)
                    penalty += max(0.0, 18.0 - improvement)
            return penalty

        def dfs(index: int, left: Counter[str], chosen: list[dict[str, Any]], holder_counts: Counter[str], total: float) -> None:
            nonlocal best_score, best_rows
            # Small bonus for concentrating a real build; do not force concentration when the
            # second/third item is materially worse for that holder.
            bundle_bonus = sum(3.0 if count == 2 else 7.0 if count >= 3 else 0.0 for count in holder_counts.values())
            diversity_penalty = max(0, len(holder_counts) - 1) * 1.2
            objective = (
                total
                + bundle_bonus
                - diversity_penalty
                - opportunity_penalty(left, chosen)
                - stability_penalty(left, chosen)
            )
            if objective > best_score:
                best_score = objective
                best_rows = list(chosen)
            if index >= len(candidates) or len(chosen) >= limit:
                return
            for candidate_index in range(index, len(candidates)):
                row = candidates[candidate_index]
                holder_id = str(row.get("holderId") or "")
                item_id = str(row.get("itemId") or "")
                if not holder_id or holder_counts[holder_id] >= 3:
                    continue
                if any(
                    str(previous.get("itemId") or "") == item_id
                    and str(previous.get("holderId") or "") == holder_id
                    for previous in chosen
                ):
                    continue
                pair_adjustment = 0.0
                incompatible = False
                for previous in chosen:
                    previous_item = str(previous.get("itemId") or "")
                    previous_holder = str(previous.get("holderId") or "")
                    if not previous_item:
                        continue
                    pair_holder = holder_id if holder_id == previous_holder else None
                    if not self._items_can_coexist(pair_holder, item_id, previous_item):
                        incompatible = True
                        break
                    if pair_holder:
                        compatibility, _, _, _, _ = self._item_pair_compatibility(
                            pair_holder,
                            item_id,
                            previous_item,
                        )
                        # Co-occurrence is deliberately a material, but not dominant, part of
                        # the objective. It should decide between similarly strong item slams.
                        pair_adjustment += (compatibility - 0.5) * 18.0
                if incompatible:
                    continue
                recipe = list(row.get("recipe") or [])
                need = Counter(recipe)
                if any(left[component] < amount for component, amount in need.items()):
                    continue
                next_left = left.copy()
                next_left.subtract(need)
                next_counts = holder_counts.copy()
                next_counts[holder_id] += 1
                # The old solver summed absolute item scores, which made *any* additional item
                # look beneficial and allowed two mediocre slams to beat one critical carry
                # item. Score only the value above holding the components, then explicitly
                # reward items that are recurrent core pieces for the intended late holder.
                marginal = float(row["score"]) - hold_threshold
                marginal += float(row.get("corePriority") or 0.0) * 26.0
                if stable_item_counts[item_id]:
                    # Item-level stability matters more than holder stability: holder can
                    # legitimately change after the board context changes, but the craft
                    # itself should remain visible when the newly added component is unrelated.
                    # Use a meaningful hysteresis margin: a newly unlocked side-grade must be
                    # materially better before it can evict an already-valid recommendation.
                    anchor_count = stable_item_counts[item_id]
                    marginal += 12.0 + min(2, max(0, anchor_count - 1)) * 2.0
                    if stable_anchor_counts[(item_id, holder_id)]:
                        marginal += min(2, stable_anchor_counts[(item_id, holder_id)]) * 2.0
                dfs(
                    candidate_index + 1,
                    next_left,
                    chosen + [row],
                    next_counts,
                    total + marginal + pair_adjustment,
                )

        if history_seed:
            # Same comp/level/owned board + additive component bag: preserve the exact previous
            # backend recommendation and only optimize how to spend the newly available
            # remainder. This is the only way to guarantee no visible item disappears during
            # incremental input, while reset/removal/comp changes still trigger a full solve.
            history_total = sum(
                (float(row.get("score") or 0.0) - hold_threshold)
                + float(row.get("corePriority") or 0.0) * 26.0
                for row in history_seed
            )
            best_score = -1e9
            dfs(0, history_left, history_seed, history_holder_counts, history_total)
        else:
            dfs(0, inventory, [], Counter(), 0.0)
        return best_rows

    def _component_fit(
        self,
        cluster: dict[str, Any],
        components: list[str],
        early_board: list[str],
        final_board: list[str],
        craftable_items: list[dict[str, Any]] | None = None,
    ) -> tuple[float, float]:
        if len(components) < 2:
            return 0.5, 0.0
        ranked: list[dict[str, Any]] = []
        for item in craftable_items if craftable_items is not None else self._craftable_completed_items(components):
            recipe = list(item.get("composition") or [])
            early_holder, early_score, _ = self._best_holder(cluster, item, early_board)
            final_holder, final_score, _ = self._best_holder(cluster, item, final_board)
            if not early_holder and not final_holder:
                continue
            global_item, global_ev = self._opgg_item_strength(str(item["id"]))
            score = max(early_score, final_score * 0.92) * 0.76 + global_item * global_ev * 0.16 + max(early_score, final_score) * 0.08
            holder_id = early_holder if early_score >= final_score * 0.92 else final_holder
            ranked.append({
                "score": clamp(score),
                "recipe": recipe,
                "itemId": str(item["id"]),
                "holderId": holder_id,
            })
        ranked.sort(key=lambda value: float(value["score"]), reverse=True)
        selected = self._select_craft_set(ranked, components, limit=3, cluster=cluster)
        if not selected:
            return 0.28, 0.0
        strength = sum(float(row["score"]) for row in selected) / len(selected)
        craft_slots = max(1, min(3, len(components) // 2))
        coverage = len(selected) / craft_slots
        return clamp(strength * 0.82 + coverage * 0.18), clamp(coverage)

    def _build_item_plan(
        self,
        cluster: dict[str, Any],
        components: list[str],
        early_board: list[str],
        final_board: list[str],
        transition_path: list[dict[str, Any]] | None = None,
        craftable_items: list[dict[str, Any]] | None = None,
        previous_item_plan: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if len(components) < 2:
            return []
        craftable = craftable_items if craftable_items is not None else self._craftable_completed_items(components)
        if not craftable:
            return []

        tempo_names = {
            "Sunfire Cape", "Gargoyle Stoneplate", "Warmog's Armor", "Guinsoo's Rageblade",
            "Spear of Shojin", "Infinity Edge", "Ionic Spark", "Evenshroud", "Red Buff",
        }
        mid_board = unique(early_board + [unit for unit in final_board if int(self.catalog.champion_by_id.get(unit, {}).get("cost") or 9) <= 4])
        if transition_path:
            future = [row for row in transition_path if int(row.get("level") or 0) > len(early_board)]
            if future:
                midpoint = min(future, key=lambda row: abs(int(row.get("level") or 0) - 6))
                mid_board = list(midpoint.get("units") or mid_board)
        stages = [
            ("opener", "2-1 → 2-5", early_board, 0.10),
            ("mid", "3-2 → 4-1", mid_board, 0.05),
            ("late", "4-2+", final_board, 0.0),
        ]
        output: list[dict[str, Any]] = []
        for stage_id, label, candidates, tempo_bonus in stages:
            ranked: list[dict[str, Any]] = []
            for item in craftable:
                final_holder, final_holder_score, final_reason = self._best_holder(cluster, item, final_board)
                item_live, item_count, _ = self._item_stat(cluster, str(item["id"]))
                opgg_item, opgg_item_ev = self._opgg_item_strength(str(item["id"]))
                tempo = tempo_bonus if str(item.get("nameEn")) in tempo_names else 0.0
                # Keep multiple holder alternatives per craft so the joint solver can form a
                # coherent 2/3-item carry/tank build instead of accepting each item's isolated
                # argmax and spreading all items across the board.
                for holder_id, holder_score, holder_reason in self._rank_holders(cluster, item, candidates)[:3]:
                    transfer = 0.04 if final_holder and final_holder != holder_id and stage_id != "late" else 0.0
                    signals: list[tuple[float, float]] = [
                        (holder_score, 0.52),
                        (item_live, 0.20),
                        (evidence(item_count), 0.10),
                        (final_holder_score, 0.12),
                    ]
                    if opgg_item_ev > 0:
                        signals.extend([(opgg_item, 0.07), (opgg_item_ev, 0.03)])
                    denominator = sum(weight for _, weight in signals)
                    score = sum(value * weight for value, weight in signals) / denominator + tempo + transfer
                    core_holder = final_holder or holder_id
                    core_priority = self._item_core_priority(cluster, str(item["id"]), core_holder)
                    ranked.append({
                        "stage": stage_id,
                        "stageLabel": label,
                        "itemId": item["id"],
                        "holderId": holder_id,
                        "finalHolderId": final_holder,
                        "score": round(clamp(score) * 100, 2),
                        "reason": holder_reason,
                        "transferReason": final_reason if final_holder and final_holder != holder_id else "",
                        "sampleCount": item_count,
                        "recipe": list(item.get("composition") or []),
                        "emblemTraitId": self._emblem_trait_id(item),
                        "corePriority": core_priority,
                        "coreHolderId": core_holder,
                    })
            # Core items must survive candidate pruning even when their isolated slam score is
            # slightly lower. The exact solver already prices component opportunity cost; give
            # it a chance to see recurrent BIS pieces instead of pruning them before search.
            ranked.sort(
                key=lambda value: float(value["score"]) + float(value.get("corePriority") or 0.0) * 10.0,
                reverse=True,
            )
            selected = self._select_stage_item_set(
                ranked,
                components,
                limit=3,
                cluster=cluster,
                stage_id=stage_id,
                previous_rows=[
                    row for row in (previous_item_plan or [])
                    if str(row.get("stage") or "") == stage_id
                ],
            )
            for row in selected:
                row.pop("recipe", None)
            output.extend(selected)

        # Add best observed late 3-item builds for carries in this comp. This is different from
        # craft-now advice: it tells the user what the holder should converge to later.
        late_builds: list[dict[str, Any]] = []
        final_set = set(final_board)
        build_rows = [row for row in cluster.get("builds") or [] if row.get("unit") in final_set]
        for row in build_rows:
            count = int(row.get("count") or 0)
            observed = placement_strength(float(row.get("avg") or 0))
            learned = row.get("ml")
            model_std = float(row.get("mlStd") or 0.0)
            model_agreement = clamp(1.0 - model_std / 0.075, 0.72, 1.0)
            score = observed * 0.50 + evidence(count) * 0.25 + (float(learned) if learned is not None else observed) * 0.25
            item_ids = list(row.get("items") or [])
            pair_scores: list[float] = []
            valid_build = True
            for left_index, left in enumerate(item_ids):
                for right in item_ids[left_index + 1 :]:
                    if not self._items_can_coexist(str(row.get("unit") or ""), str(left), str(right)):
                        valid_build = False
                        break
                    pair_scores.append(
                        self._item_pair_compatibility(
                            str(row.get("unit") or ""),
                            str(left),
                            str(right),
                        )[0]
                    )
                if not valid_build:
                    break
            if not valid_build:
                continue
            pair_coherence = sum(pair_scores) / len(pair_scores) if pair_scores else 0.5
            score *= 0.82 + pair_coherence * 0.18
            late_builds.append({
                "stage": "bis",
                "stageLabel": "Board cuối",
                "holderId": row["unit"],
                "itemIds": item_ids,
                "score": round(score * model_agreement * 100, 2),
                "sampleCount": count,
                "avgPlacement": float(row.get("avg") or 0),
                "modelDisagreement": round(model_std * 100, 2),
            })
        late_builds.sort(key=lambda value: value["score"], reverse=True)
        output.extend(late_builds[:4])
        return output

    def recommend(
        self,
        level: int,
        owned_ids: list[str],
        components: list[str],
        target_comp_id: str | None = None,
        *,
        previous_level: int | None = None,
        previous_comp_id: str | None = None,
        previous_owned_ids: list[str] | None = None,
        previous_components: list[str] | None = None,
        previous_item_plan: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        level = max(3, min(10, int(level)))
        owned = {value for value in owned_ids if value in self.catalog.champion_by_id}
        craftable_items = self._craftable_completed_items(components)
        candidates: list[dict[str, Any]] = []
        for cluster in self.clusters:
            final_board, final_score, final_option, final_level = self._best_final(cluster)
            if len(final_board) < 4:
                continue
            early_board, early_fit, early_option = self._best_early(cluster, level, owned)
            if not early_board:
                early_board = self._fallback_early_board(level, owned)
                early_fit = self._heuristic_board_score(early_board, owned, True) * 0.72
            component_fit, component_coverage = self._component_fit(
                cluster,
                components,
                early_board[:level],
                final_board,
                craftable_items,
            )
            component_signal_confidence = clamp(
                component_coverage * min(1.0, len(components) / 4.0)
            )
            own_final = len(owned.intersection(final_board)) / max(1, len(owned)) if owned else 0.5
            overall = cluster.get("overall") or {}
            overall_strength = placement_strength(float(overall.get("avg") or 0))
            sample = evidence(int(overall.get("count") or 0))
            opgg_strength, opgg_ev, opgg_similarity, opgg_row = self._opgg_comp_match(final_board)
            pro_strength, pro_ev = self._high_elo_board_prior(final_board)
            weighted_signals = [
                (final_score, 0.34),
                (early_fit, 0.28),
                (own_final, 0.10),
                (overall_strength, 0.09),
                (sample, 0.07),
            ]
            if len(components) >= 2 and component_signal_confidence > 0:
                # A couple of awkward components should not redirect the whole comp ranking.
                # Item fit becomes material only when the bag can actually make useful slams.
                weighted_signals.extend([
                    (component_fit, 0.10 * component_signal_confidence),
                    (component_coverage, 0.025 * component_signal_confidence),
                ])
            if opgg_row is not None:
                weighted_signals.extend([(opgg_strength, 0.09), (opgg_ev * opgg_similarity, 0.03)])
            if pro_ev > 0:
                weighted_signals.extend([(pro_strength, 0.035), (pro_ev, 0.015)])
            base_denominator = sum(weight for _, weight in weighted_signals)
            base_numerator = sum(value * weight for value, weight in weighted_signals)
            # Transition is deliberately neutral during the cheap first pass. Only the best
            # candidates pay for the dynamic-programming path below.
            transition_fit = 0.5
            transition_path: list[dict[str, Any]] = []
            cluster_score = (base_numerator + transition_fit * 0.07) / (base_denominator + 0.07)
            comparable = [final_score, early_fit, overall_strength]
            if opgg_row is not None:
                comparable.append(opgg_strength)
            mean = sum(comparable) / len(comparable)
            disagreement = math.sqrt(sum((value - mean) ** 2 for value in comparable) / len(comparable))
            final_ev = evidence(int(final_option.get("count") or 0)) if final_option else 0.0
            early_ev = evidence(int(early_option.get("count") or 0)) if early_option else 0.0
            agreement = clamp(1.0 - disagreement / 0.32)
            model_std = max(
                float(final_option.get("mlStd") or 0.0) if final_option else 0.0,
                float(early_option.get("mlStd") or 0.0) if early_option else 0.0,
            )
            model_agreement = clamp(1.0 - model_std / 0.075, 0.68, 1.0)
            confidence = clamp(
                (0.18 + final_ev * 0.25 + early_ev * 0.18 + sample * 0.14 + opgg_ev * opgg_similarity * 0.20)
                * (0.72 + agreement * 0.28)
                * model_agreement
            )
            if len(components) >= 2 and component_signal_confidence > 0:
                confidence = clamp(confidence * (0.98 + component_signal_confidence * 0.02))
            reasons: list[str] = []
            hits = len(owned.intersection(early_board))
            if hits:
                reasons.append(f"Giữ được {hits} tướng bạn đang có ở board def")
            if early_option and int(early_option.get("count") or 0) >= 5:
                reasons.append(f"Opener có {int(early_option.get('count') or 0)} mẫu live")
            if final_option and int(final_option.get("count") or 0) >= 5:
                reasons.append(f"Board cuối avg {float(final_option.get('avg') or 0):.2f} / {int(final_option.get('count') or 0)} mẫu")
            if opgg_row is not None and opgg_similarity >= 0.62:
                reasons.append(f"OP.GG cross-check {int(opgg_similarity * 100)}% board · {int(opgg_row.get('games') or 0)} mẫu")
            if len(components) >= 2 and component_coverage > 0:
                reasons.append(f"Đồ hiện tại fit {int(component_fit * 100)}% · dùng được {int(component_coverage * 100)}% slot ghép")
            elif len(components) >= 2:
                reasons.append("Chưa có slam đủ tốt; giữ component không làm lệch hướng đội hình")
            candidates.append({
                "cluster": cluster,
                "score": cluster_score,
                "earlyBoard": early_board[:level],
                "finalBoard": final_board,
                "finalLevel": final_level,
                "reasons": reasons,
                "confidence": confidence,
                "crossSource": opgg_row is not None,
                "modelStd": model_std,
                "componentFit": component_fit,
                "componentCoverage": component_coverage,
                "componentSignalConfidence": component_signal_confidence,
                "transitionFit": transition_fit,
                "transitionPath": transition_path,
                "baseNumerator": base_numerator,
                "baseDenominator": base_denominator,
            })

        # Transition DP was the dominant runtime cost. A 7% transition weight cannot
        # realistically promote a weak comp from the bottom of 53 clusters into the top six,
        # so evaluate it exactly only for a generous preliminary shortlist.
        candidates.sort(key=lambda value: value["score"], reverse=True)
        transition_candidates = list(candidates[:18])
        if target_comp_id is not None:
            target_candidate = next(
                (candidate for candidate in candidates if str(candidate["cluster"].get("id")) == str(target_comp_id)),
                None,
            )
            if target_candidate is not None and all(candidate is not target_candidate for candidate in transition_candidates):
                transition_candidates.append(target_candidate)
        for candidate in transition_candidates:
            transition_path, transition_fit = self._transition_path(
                candidate["cluster"],
                level,
                owned,
                list(candidate["earlyBoard"]),
                list(candidate["finalBoard"]),
                int(candidate["finalLevel"]),
            )
            candidate["transitionPath"] = transition_path
            candidate["transitionFit"] = transition_fit
            candidate["score"] = (
                float(candidate["baseNumerator"]) + transition_fit * 0.07
            ) / (float(candidate["baseDenominator"]) + 0.07)
        candidates.sort(key=lambda value: value["score"], reverse=True)
        stable_state = (
            previous_level == level
            and set(previous_owned_ids or []) == owned
            and all(
                Counter(components)[component] >= amount
                for component, amount in Counter(previous_components or []).items()
            )
        )
        if stable_state and target_comp_id is None and previous_comp_id and candidates:
            previous_candidate = next(
                (candidate for candidate in candidates if str(candidate["cluster"].get("id")) == str(previous_comp_id)),
                None,
            )
            # Adding unrelated components should not cause a tiny item-fit delta to flicker the
            # recommended comp. A clearly stronger comp still replaces it immediately.
            if previous_candidate is not None and float(previous_candidate["score"]) >= float(candidates[0]["score"]) - 0.018:
                candidates.remove(previous_candidate)
                candidates.insert(0, previous_candidate)
                previous_candidate["reasons"].append("Giữ hướng cũ: thay đổi hiện tại chưa đủ lớn để pivot")
        top = next(
            (candidate for candidate in candidates if str(candidate["cluster"].get("id")) == str(target_comp_id)),
            candidates[0] if candidates else None,
        )
        if top is None:
            early = self._fallback_early_board(level, owned)
            return {
                "earlyBoardIds": early,
                "earlyTraits": self.active_traits(early),
                "buyNextIds": [],
                "comps": [],
                "itemPlan": [],
                "model": self.model_status(),
                "data": self.data_status(),
            }

        top_final = list(top["finalBoard"])
        transition_units = [
            unit
            for row in list(top.get("transitionPath") or [])[1:]
            for unit in list(row.get("units") or [])
        ]
        buy_next = unique(transition_units + list(top["earlyBoard"]) + top_final)
        buy_next = [value for value in buy_next if value not in owned]
        early_set = set(top["earlyBoard"])
        final_set = set(top_final)
        def buy_priority(value: str) -> tuple[float, int]:
            unit = self.catalog.champion_by_id[value]
            cost = int(unit.get("cost") or 9)
            live_strength, live_ev = self._opgg_unit_strength(value)
            pro_strength, pro_ev = self._high_elo_unit_prior(value)
            stage_fit = 2.0 if value in early_set else 1.0 if value in final_set and cost <= max(3, level) else 0.35
            affordability = 1.0 if cost <= max(2, min(4, level - 1)) else 0.35
            score = stage_fit + live_strength * live_ev * 0.38 + pro_strength * pro_ev * 0.16 + affordability * 0.25
            return (-score, cost)
        buy_next.sort(key=buy_priority)

        # Reroll/star models answer a different question than board strength: what upgrade
        # state this observed comp normally needs. Batch all six candidates so TensorFlow is
        # invoked once per ensemble instead of once per champion.
        shown_candidates = list(candidates[:6])
        if all(candidate is not top for candidate in shown_candidates):
            shown_candidates = [top] + [candidate for candidate in shown_candidates if candidate is not top][:5]
        reroll_predictions = self.reroll_ranker.score_many([
            {
                "units": list(candidate["finalBoard"]),
                "level": int(candidate["finalLevel"]),
                "sample_kind": "reroll_board",
            }
            for candidate in shown_candidates
        ])
        star_rows: list[dict[str, Any]] = []
        star_owner: list[int] = []
        for candidate_index, candidate in enumerate(shown_candidates):
            for unit_id in candidate["finalBoard"]:
                star_rows.append({
                    "board": list(candidate["finalBoard"]),
                    "unit": unit_id,
                    "level": int(candidate["finalLevel"]),
                })
                star_owner.append(candidate_index)
        star_predictions = self.star_ranker.predict_many(star_rows)
        star_by_candidate: list[list[dict[str, Any]]] = [[] for _ in shown_candidates]
        for owner, row, prediction in zip(star_owner, star_rows, star_predictions, strict=True):
            if prediction is None:
                continue
            distribution = list(prediction.get("distribution") or [0.0, 1.0, 0.0])
            star_by_candidate[owner].append({
                "unitId": str(row["unit"]),
                "stars": int(prediction.get("stars") or 2),
                "confidence": round(float(prediction.get("confidence") or 0.0) * 100, 1),
                "threeStarProbability": round(float(distribution[2] if len(distribution) > 2 else 0.0) * 100, 1),
            })

        comps: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(shown_candidates):
            rank = next(
                (index + 1 for index, ranked_candidate in enumerate(candidates) if ranked_candidate is candidate),
                candidate_index + 1,
            )
            cluster = candidate["cluster"]
            avg = float((cluster.get("overall") or {}).get("avg") or 0)
            tier = "S" if avg and avg <= 3.45 else "A" if avg and avg <= 4.05 else "B"
            name_parts = list(cluster.get("nameParts") or [])
            carry_ids = [
                str(part.get("name")) for part in name_parts
                if isinstance(part, dict) and part.get("type") == "unit" and str(part.get("name")) in self.catalog.champion_by_id
            ]
            if not carry_ids:
                carry_ids = [
                    row["unit"] for row in cluster.get("unitStats") or []
                    if row.get("unit") in candidate["finalBoard"]
                ][:3]
            reroll_score = float(reroll_predictions[candidate_index] or 0.0)
            star_targets = star_by_candidate[candidate_index]
            # When the board classifier says reroll, a moderately strong P(3★) should be
            # surfaced as the goal even if the imbalanced star classifier's argmax is still
            # 2★. This keeps the decision driven by both trained models, not champion names.
            for target in star_targets:
                champion = self.catalog.champion_by_id.get(str(target["unitId"]), {})
                if (
                    reroll_score >= 0.50
                    and int(champion.get("cost") or 9) <= 3
                    and float(target.get("threeStarProbability") or 0.0) >= 24.0
                ):
                    target["stars"] = 3
            reroll_core = [
                target for target in star_targets
                if int(target.get("stars") or 2) >= 3
                and int(self.catalog.champion_by_id.get(str(target["unitId"]), {}).get("cost") or 9) <= 3
            ]
            primary_reroll_core = sorted(
                reroll_core,
                key=lambda row: float(row.get("threeStarProbability") or 0.0),
                reverse=True,
            )[:3]
            roll_level = int(candidate["finalLevel"])
            if primary_reroll_core:
                core_cost = max(
                    int(self.catalog.champion_by_id.get(str(target["unitId"]), {}).get("cost") or 1)
                    for target in primary_reroll_core
                )
                roll_level = {1: 5, 2: 6, 3: 7}.get(core_cost, roll_level)
            leveling = f"Hướng tới level {candidate['finalLevel']}"
            if reroll_score >= 0.50 and primary_reroll_core:
                core_names = [
                    str(self.catalog.champion_by_id.get(str(target["unitId"]), {}).get("name") or target["unitId"])
                    for target in primary_reroll_core
                ]
                leveling = f"Reroll Lv.{roll_level} · ưu tiên 3★ {' / '.join(core_names)}"
            comps.append({
                "id": str(cluster.get("id")),
                "rank": rank,
                "name": self.comp_name(cluster),
                "tier": tier,
                "score": round(float(candidate["score"]) * 100, 2),
                "confidence": round(float(candidate["confidence"]) * 100, 1),
                "uncertainty": round((1.0 - float(candidate["confidence"])) * 100, 1),
                "crossSource": bool(candidate["crossSource"]),
                "modelDisagreement": round(float(candidate["modelStd"]) * 100, 2),
                "componentFit": round(float(candidate["componentFit"]) * 100, 1),
                "componentFitConfidence": round(float(candidate["componentSignalConfidence"]) * 100, 1),
                "transitionFit": round(float(candidate["transitionFit"]) * 100, 1),
                "avgPlacement": avg or None,
                "games": int((cluster.get("overall") or {}).get("count") or 0),
                "pickRate": float((cluster.get("overall") or {}).get("pick") or 0),
                "earlyBoardIds": candidate["earlyBoard"],
                "boardIds": candidate["finalBoard"],
                "carryIds": carry_ids[:4],
                "reroll": reroll_score >= 0.50,
                "rerollScore": round(reroll_score * 100, 1),
                "rollLevel": roll_level if reroll_score >= 0.50 else None,
                "starTargets": star_targets,
                "positioning": self._position_board(cluster, list(candidate["finalBoard"])) if candidate is top else [],
                "activeTraits": self.active_traits(candidate["finalBoard"]),
                "matchReasons": candidate["reasons"],
                "leveling": leveling,
                "transitionPath": [
                    {
                        "level": int(row.get("level") or 0),
                        "boardIds": list(row.get("units") or []),
                        "avgPlacement": row.get("avgPlacement"),
                        "games": int(row.get("games") or 0),
                    }
                    for row in candidate.get("transitionPath") or []
                ],
            })

        item_plan = self._build_item_plan(
            top["cluster"],
            components,
            list(top["earlyBoard"]),
            top_final,
            list(top.get("transitionPath") or []),
            craftable_items,
            previous_item_plan=(
                previous_item_plan
                if previous_level == level
                and str(previous_comp_id or "") == str(top["cluster"].get("id") or "")
                and set(previous_owned_ids or []) == owned
                and all(
                    Counter(components)[component] >= amount
                    for component, amount in Counter(previous_components or []).items()
                )
                else []
            ),
        )
        opener_emblems = [
            str(row.get("emblemTraitId"))
            for row in item_plan
            if row.get("stage") == "opener" and row.get("emblemTraitId")
        ]
        late_emblems = [
            str(row.get("emblemTraitId"))
            for row in item_plan
            if row.get("stage") == "late" and row.get("emblemTraitId")
        ]
        for comp in comps:
            if comp.get("id") == str(top["cluster"].get("id")):
                comp["activeTraits"] = self.active_traits(top_final, late_emblems)
        return {
            "earlyBoardIds": list(top["earlyBoard"]),
            "earlyPositioning": self._position_board(top["cluster"], list(top["earlyBoard"])),
            "earlyTraits": self.active_traits(list(top["earlyBoard"]), opener_emblems),
            "buyNextIds": buy_next[:10],
            "comps": comps,
            "itemPlan": item_plan,
            "model": self.model_status(),
            "data": self.data_status(),
        }

    def model_status(self) -> dict[str, Any]:
        return {
            "boardAvailable": self.board_ranker.available and self.board_ranker.error is None,
            "board": self.board_ranker.metadata(),
            "boardRuntimeValueReliability": round(self._ranker_value_reliability(self.board_ranker), 3),
            "boardError": self.board_ranker.error,
            "itemAvailable": self.item_ranker.available and self.item_ranker.error is None,
            "item": self.item_ranker.metadata(),
            "itemRuntimeValueReliability": round(self._ranker_value_reliability(self.item_ranker), 3),
            "itemError": self.item_ranker.error,
            "itemAffinityAvailable": self.item_affinity_ranker.available and self.item_affinity_ranker.error is None,
            "itemAffinity": self.item_affinity_ranker.metadata(),
            "itemAffinityError": self.item_affinity_ranker.error,
            "itemPairAvailable": self.item_pair_ranker.available and self.item_pair_ranker.error is None,
            "itemPair": self.item_pair_ranker.metadata(),
            "itemPairError": self.item_pair_ranker.error,
            "positionAvailable": self.position_ranker.available and self.position_ranker.error is None,
            "position": self.position_ranker.metadata(),
            "positionError": self.position_ranker.error,
            "rerollAvailable": self.reroll_ranker.available and self.reroll_ranker.error is None,
            "reroll": self.reroll_ranker.metadata(),
            "rerollError": self.reroll_ranker.error,
            "starAvailable": self.star_ranker.available and self.star_ranker.error is None,
            "star": self.star_ranker.metadata(),
            "starError": self.star_ranker.error,
        }

    def data_status(self) -> dict[str, Any]:
        return {
            "set": self.snapshot.get("set"),
            "patch": self.snapshot.get("patch"),
            "queue": self.snapshot.get("queue"),
            "generatedAt": self.snapshot.get("generatedAt"),
            "clusterId": self.snapshot.get("clusterId"),
            "clusters": len(self.clusters),
            "sources": self.health.get("sources", {}),
            "trainingSamples": self.health.get("total", 0),
            "crossSourceRows": self.health.get("crossSourceRows", 0),
            "opggGames24h": self.opgg.get("games24h", 0),
            "highEloUnitPriors": self.health.get("highEloUnitPriors", 0),
            "highEloItemHolderPriors": self.health.get("highEloItemHolderPriors", 0),
        }

