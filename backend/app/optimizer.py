from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from backend.ml.ranker import ItemRanker, LearnedRanker

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
        self.snapshot = self._load_json(SNAPSHOT_PATH, {"clusters": []})
        self.opgg = self._load_json(OPGG_PATH, {})
        self.high_elo = self._load_json(HIGH_ELO_PATH, {"units": {}, "itemHolders": {}})
        self.health = self._load_json(HEALTH_PATH, {"sources": {}, "total": 0})
        self.clusters: list[dict[str, Any]] = list(self.snapshot.get("clusters") or [])
        self._index_opgg()
        self._precompute_ml()

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
        self._index_opgg()
        self.board_ranker = LearnedRanker(self.catalog)
        self.item_ranker = ItemRanker(self.catalog)
        self._precompute_ml()

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
                destination["ml"] = score
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
                destination["ml"] = score
        for destination, row in zip(item_destinations, item_rows, strict=True):
            uncertainty = self.item_ranker.uncertainty(
                units=list(row["units"]),
                items=list(row["items"]),
                level=8,
                sample_kind="item_holder_build",
            )
            if uncertainty is not None:
                destination["mlStd"] = uncertainty

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

    def active_traits(self, unit_ids: list[str]) -> list[dict[str, Any]]:
        counts: Counter[str] = Counter()
        for unit_id in unit_ids:
            champion = self.catalog.champion_by_id.get(unit_id)
            if champion:
                counts.update(champion.get("traits") or [])
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
            return best[1], best[0], best[2], best[3]
        centroid = self._known_units(list(cluster.get("centroidUnits") or []))[:8]
        return centroid, self._heuristic_board_score(centroid, set(), False), None, len(centroid) or 8

    def _observed_level_boards(self, cluster: dict[str, Any], level: int) -> list[dict[str, Any]]:
        """Merge early/final observed boards for one level and keep only strong distinct nodes."""
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
        return sorted(output.values(), key=lambda value: float(value["score"]), reverse=True)[:24]

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

    @staticmethod
    def _role_is_tank(role: str) -> bool:
        return "Tank" in role or "Fighter" in role

    @staticmethod
    def _role_is_ad(role: str) -> bool:
        return role.startswith("AD")

    @staticmethod
    def _role_is_ap(role: str) -> bool:
        return role.startswith("AP")

    def _item_role_score(self, item: dict[str, Any], holder_id: str) -> float:
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
        return clamp(score)

    @staticmethod
    def _can_craft(recipe: list[str], components: list[str]) -> bool:
        available = Counter(components)
        for component in recipe:
            if available[component] <= 0:
                return False
            available[component] -= 1
        return True

    def _item_stat(self, cluster: dict[str, Any], item_id: str, holder_id: str | None = None) -> tuple[float, int, float]:
        for row in cluster.get("itemStats") or []:
            if row.get("item") != item_id:
                continue
            if holder_id:
                for holder in row.get("units") or []:
                    if holder.get("unit") == holder_id:
                        avg = float(holder.get("avg") or 0)
                        return placement_strength(avg), int(holder.get("count") or 0), float(holder.get("placeChange") or 0)
                # Global item strength is not evidence that an arbitrary champion is a good
                # holder. Falling through here used to create false-positive holder matches.
                return 0.5, 0, 0.0
            return placement_strength(float(row.get("avg") or 0)), int(row.get("count") or 0), 0.0
        return 0.5, 0, 0.0

    def _best_holder(self, cluster: dict[str, Any], item: dict[str, Any], candidates: list[str]) -> tuple[str | None, float, str]:
        best: tuple[float, str, str] | None = None
        for holder_id in candidates:
            if holder_id not in self.catalog.champion_by_id:
                continue
            empirical, count, place_change = self._item_stat(cluster, str(item["id"]), holder_id)
            role = self._item_role_score(item, holder_id)
            sample = evidence(count)
            improvement = clamp((-place_change + 1.0) / 2.0) if count else 0.5
            opgg, opgg_ev = self._opgg_item_strength(str(item["id"]), holder_id)
            pro, pro_ev = self._high_elo_item_holder_prior(str(item["id"]), holder_id)
            signals: list[tuple[float, float]] = [
                (role, 0.38),
                (empirical, 0.29),
                (sample, 0.18),
                (improvement, 0.15),
            ]
            if opgg_ev > 0:
                signals.extend([(opgg, 0.12), (opgg_ev, 0.04)])
            if pro_ev > 0:
                signals.extend([(pro, 0.06), (pro_ev, 0.02)])
            denominator = sum(weight for _, weight in signals)
            score = sum(value * weight for value, weight in signals) / denominator
            reason = "hợp vai trò"
            if count >= 5 and opgg_ev > 0 and pro_ev > 0:
                reason = f"MetaTFT {count} mẫu + OP.GG + high-Elo xác nhận"
            elif count >= 5 and opgg_ev > 0:
                reason = f"MetaTFT {count} mẫu + OP.GG xác nhận holder"
            elif count >= 5 and pro_ev > 0:
                reason = f"MetaTFT {count} mẫu + high-Elo đang dùng"
            elif count >= 5:
                reason = f"{count} mẫu holder live, avg {8.5 - empirical * 7.5:.2f}"
            if best is None or score > best[0]:
                best = (score, holder_id, reason)
        return (best[1], best[0], best[2]) if best else (None, 0.0, "")

    def _select_craft_set(self, ranked: list[dict[str, Any]], components: list[str], limit: int = 3) -> list[dict[str, Any]]:
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
                recipe = list(row.get("recipe") or [])
                need = Counter(recipe)
                if any(left[component] < amount for component, amount in need.items()):
                    continue
                next_left = left.copy()
                next_left.subtract(need)
                dfs(candidate_index + 1, next_left, chosen + [row], total + float(row["score"]))

        dfs(0, inventory, [], 0.0)
        return best_rows

    def _component_fit(
        self,
        cluster: dict[str, Any],
        components: list[str],
        early_board: list[str],
        final_board: list[str],
    ) -> tuple[float, float]:
        if len(components) < 2:
            return 0.5, 0.0
        ranked: list[dict[str, Any]] = []
        for item in self.catalog.items:
            recipe = list(item.get("composition") or [])
            if item.get("category") != "completed" or len(recipe) != 2 or not self._can_craft(recipe, components):
                continue
            early_holder, early_score, _ = self._best_holder(cluster, item, early_board)
            final_holder, final_score, _ = self._best_holder(cluster, item, final_board)
            if not early_holder and not final_holder:
                continue
            global_item, global_ev = self._opgg_item_strength(str(item["id"]))
            score = max(early_score, final_score * 0.92) * 0.76 + global_item * global_ev * 0.16 + max(early_score, final_score) * 0.08
            ranked.append({"score": clamp(score), "recipe": recipe})
        ranked.sort(key=lambda value: float(value["score"]), reverse=True)
        selected = self._select_craft_set(ranked, components, limit=3)
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
    ) -> list[dict[str, Any]]:
        if len(components) < 2:
            return []
        craftable = [
            item for item in self.catalog.items
            if item.get("category") == "completed"
            and len(item.get("composition") or []) == 2
            and self._can_craft(list(item.get("composition") or []), components)
        ]
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
                holder_id, holder_score, holder_reason = self._best_holder(cluster, item, candidates)
                if not holder_id:
                    continue
                final_holder, final_holder_score, final_reason = self._best_holder(cluster, item, final_board)
                item_live, item_count, _ = self._item_stat(cluster, str(item["id"]))
                opgg_item, opgg_item_ev = self._opgg_item_strength(str(item["id"]))
                tempo = tempo_bonus if str(item.get("nameEn")) in tempo_names else 0.0
                transfer = 0.04 if final_holder and final_holder != holder_id and stage_id != "late" else 0.0
                signals: list[tuple[float, float]] = [
                    (holder_score, 0.48),
                    (item_live, 0.24),
                    (evidence(item_count), 0.12),
                    (final_holder_score, 0.10),
                ]
                if opgg_item_ev > 0:
                    signals.extend([(opgg_item, 0.07), (opgg_item_ev, 0.03)])
                denominator = sum(weight for _, weight in signals)
                score = sum(value * weight for value, weight in signals) / denominator + tempo + transfer
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
                })
            ranked.sort(key=lambda value: value["score"], reverse=True)
            selected = self._select_craft_set(ranked, components, limit=3)
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
            late_builds.append({
                "stage": "bis",
                "stageLabel": "Board cuối",
                "holderId": row["unit"],
                "itemIds": list(row.get("items") or []),
                "score": round(score * model_agreement * 100, 2),
                "sampleCount": count,
                "avgPlacement": float(row.get("avg") or 0),
                "modelDisagreement": round(model_std * 100, 2),
            })
        late_builds.sort(key=lambda value: value["score"], reverse=True)
        output.extend(late_builds[:4])
        return output

    def recommend(self, level: int, owned_ids: list[str], components: list[str]) -> dict[str, Any]:
        level = max(3, min(10, int(level)))
        owned = {value for value in owned_ids if value in self.catalog.champion_by_id}
        candidates: list[dict[str, Any]] = []
        for cluster in self.clusters:
            final_board, final_score, final_option, final_level = self._best_final(cluster)
            if len(final_board) < 4:
                continue
            early_board, early_fit, early_option = self._best_early(cluster, level, owned)
            if not early_board:
                early_board = self._fallback_early_board(level, owned)
                early_fit = self._heuristic_board_score(early_board, owned, True) * 0.72
            transition_path, transition_fit = self._transition_path(
                cluster,
                level,
                owned,
                early_board[:level],
                final_board,
                final_level,
            )
            component_fit, component_coverage = self._component_fit(
                cluster,
                components,
                early_board[:level],
                final_board,
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
                (transition_fit, 0.07),
            ]
            if len(components) >= 2:
                weighted_signals.extend([(component_fit, 0.12), (component_coverage, 0.03)])
            if opgg_row is not None:
                weighted_signals.extend([(opgg_strength, 0.09), (opgg_ev * opgg_similarity, 0.03)])
            if pro_ev > 0:
                weighted_signals.extend([(pro_strength, 0.035), (pro_ev, 0.015)])
            denominator = sum(weight for _, weight in weighted_signals)
            cluster_score = sum(value * weight for value, weight in weighted_signals) / denominator
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
            if len(components) >= 2:
                confidence = clamp(confidence * (0.90 + component_coverage * 0.10))
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
            if len(components) >= 2:
                reasons.append(f"Đồ hiện tại fit {int(component_fit * 100)}% · dùng được {int(component_coverage * 100)}% slot ghép")
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
                "transitionFit": transition_fit,
                "transitionPath": transition_path,
            })

        candidates.sort(key=lambda value: value["score"], reverse=True)
        top = candidates[0] if candidates else None
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

        comps: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates[:6], start=1):
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
                "transitionFit": round(float(candidate["transitionFit"]) * 100, 1),
                "avgPlacement": avg or None,
                "games": int((cluster.get("overall") or {}).get("count") or 0),
                "pickRate": float((cluster.get("overall") or {}).get("pick") or 0),
                "earlyBoardIds": candidate["earlyBoard"],
                "boardIds": candidate["finalBoard"],
                "carryIds": carry_ids[:4],
                "activeTraits": self.active_traits(candidate["finalBoard"]),
                "matchReasons": candidate["reasons"],
                "leveling": f"Hướng tới level {candidate['finalLevel']}",
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

        return {
            "earlyBoardIds": list(top["earlyBoard"]),
            "earlyTraits": self.active_traits(list(top["earlyBoard"])),
            "buyNextIds": buy_next[:10],
            "comps": comps,
            "itemPlan": self._build_item_plan(
                top["cluster"],
                components,
                list(top["earlyBoard"]),
                top_final,
                list(top.get("transitionPath") or []),
            ),
            "model": self.model_status(),
            "data": self.data_status(),
        }

    def model_status(self) -> dict[str, Any]:
        return {
            "boardAvailable": self.board_ranker.available and self.board_ranker.error is None,
            "board": self.board_ranker.metadata(),
            "boardError": self.board_ranker.error,
            "itemAvailable": self.item_ranker.available and self.item_ranker.error is None,
            "item": self.item_ranker.metadata(),
            "itemError": self.item_ranker.error,
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

