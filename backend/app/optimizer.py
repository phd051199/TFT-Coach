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
        self.health = self._load_json(HEALTH_PATH, {"sources": {}, "total": 0})
        self.clusters: list[dict[str, Any]] = list(self.snapshot.get("clusters") or [])
        self._precompute_ml()

    @staticmethod
    def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            return default

    def reload(self) -> None:
        self.snapshot = self._load_json(SNAPSHOT_PATH, {"clusters": []})
        self.health = self._load_json(HEALTH_PATH, {"sources": {}, "total": 0})
        self.clusters = list(self.snapshot.get("clusters") or [])
        self.board_ranker = LearnedRanker(self.catalog)
        self.item_ranker = ItemRanker(self.catalog)
        self._precompute_ml()

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
        return clamp(raw / 12.5)

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
        tags = set(item.get("tags") or [])
        role = str(holder.get("role") or "")
        score = 0.18
        if "tank" in tags and self._role_is_tank(role):
            score += 0.52
        if "attack" in tags and self._role_is_ad(role):
            score += 0.52
        if "magic" in tags and self._role_is_ap(role):
            score += 0.52
        if "stacking" in tags and any(token in role for token in ("Carry", "Caster")):
            score += 0.10
        name = str(item.get("nameEn") or "").lower()
        if any(token in name for token in ("guinsoo", "kraken", "red buff")) and self._role_is_ad(role):
            score += 0.16
        if any(token in name for token in ("shojin", "blue buff", "archangel")) and self._role_is_ap(role):
            score += 0.16
        if any(token in name for token in ("warmog", "gargoyle", "bramble", "dragon's claw")) and self._role_is_tank(role):
            score += 0.16
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
            score = role * 0.43 + empirical * 0.30 + sample * 0.17 + improvement * 0.10
            reason = "hợp vai trò"
            if count >= 5:
                reason = f"{count} mẫu holder live, avg {8.5 - empirical * 7.5:.2f}"
            if best is None or score > best[0]:
                best = (score, holder_id, reason)
        return (best[1], best[0], best[2]) if best else (None, 0.0, "")

    def _build_item_plan(
        self,
        cluster: dict[str, Any],
        components: list[str],
        early_board: list[str],
        final_board: list[str],
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
        stages = [
            ("opener", "2-1 → 2-5", early_board, 0.16),
            ("mid", "3-2 → 4-1", unique(early_board + [unit for unit in final_board if int(self.catalog.champion_by_id.get(unit, {}).get("cost") or 9) <= 4]), 0.08),
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
                tempo = tempo_bonus if str(item.get("nameEn")) in tempo_names else 0.0
                transfer = 0.10 if final_holder and final_holder != holder_id and stage_id != "late" else 0.0
                score = holder_score * 0.50 + item_live * 0.24 + evidence(item_count) * 0.11 + tempo + final_holder_score * 0.10 + transfer
                ranked.append({
                    "stage": stage_id,
                    "stageLabel": label,
                    "itemId": item["id"],
                    "holderId": holder_id,
                    "finalHolderId": final_holder,
                    "score": round(score * 100, 2),
                    "reason": holder_reason,
                    "transferReason": final_reason if final_holder and final_holder != holder_id else "",
                    "sampleCount": item_count,
                })
            ranked.sort(key=lambda value: value["score"], reverse=True)
            output.extend(ranked[:3])

        # Add best observed late 3-item builds for carries in this comp. This is different from
        # craft-now advice: it tells the user what the holder should converge to later.
        late_builds: list[dict[str, Any]] = []
        final_set = set(final_board)
        build_rows = [row for row in cluster.get("builds") or [] if row.get("unit") in final_set]
        ml_rows = [
            {"units": [row["unit"]], "items": list(row.get("items") or []), "sample_kind": "item_holder_build", "level": 8}
            for row in build_rows
        ]
        ml_scores = self.item_ranker.score_many(ml_rows)
        for row, learned in zip(build_rows, ml_scores, strict=True):
            count = int(row.get("count") or 0)
            observed = placement_strength(float(row.get("avg") or 0))
            score = observed * 0.50 + evidence(count) * 0.25 + (float(learned) if learned is not None else observed) * 0.25
            late_builds.append({
                "stage": "bis",
                "stageLabel": "Board cuối",
                "holderId": row["unit"],
                "itemIds": list(row.get("items") or []),
                "score": round(score * 100, 2),
                "sampleCount": count,
                "avgPlacement": float(row.get("avg") or 0),
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
            own_final = len(owned.intersection(final_board)) / max(1, len(owned)) if owned else 0.5
            overall = cluster.get("overall") or {}
            overall_strength = placement_strength(float(overall.get("avg") or 0))
            sample = evidence(int(overall.get("count") or 0))
            cluster_score = final_score * 0.40 + early_fit * 0.32 + own_final * 0.10 + overall_strength * 0.10 + sample * 0.08
            reasons: list[str] = []
            hits = len(owned.intersection(early_board))
            if hits:
                reasons.append(f"Giữ được {hits} tướng bạn đang có ở board def")
            if early_option and int(early_option.get("count") or 0) >= 5:
                reasons.append(f"Opener có {int(early_option.get('count') or 0)} mẫu live")
            if final_option and int(final_option.get("count") or 0) >= 5:
                reasons.append(f"Board cuối avg {float(final_option.get('avg') or 0):.2f} / {int(final_option.get('count') or 0)} mẫu")
            candidates.append({
                "cluster": cluster,
                "score": cluster_score,
                "earlyBoard": early_board[:level],
                "finalBoard": final_board,
                "finalLevel": final_level,
                "reasons": reasons,
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
        buy_next = unique(list(top["earlyBoard"]) + top_final)
        buy_next = [value for value in buy_next if value not in owned]
        buy_next.sort(key=lambda value: (0 if value in top_final else 1, int(self.catalog.champion_by_id[value].get("cost") or 9)))

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
                "avgPlacement": avg or None,
                "games": int((cluster.get("overall") or {}).get("count") or 0),
                "pickRate": float((cluster.get("overall") or {}).get("pick") or 0),
                "earlyBoardIds": candidate["earlyBoard"],
                "boardIds": candidate["finalBoard"],
                "carryIds": carry_ids[:4],
                "activeTraits": self.active_traits(candidate["finalBoard"]),
                "matchReasons": candidate["reasons"],
                "leveling": f"Hướng tới level {candidate['finalLevel']}",
            })

        return {
            "earlyBoardIds": list(top["earlyBoard"]),
            "earlyTraits": self.active_traits(list(top["earlyBoard"])),
            "buyNextIds": buy_next[:10],
            "comps": comps,
            "itemPlan": self._build_item_plan(top["cluster"], components, list(top["earlyBoard"]), top_final),
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
        }

