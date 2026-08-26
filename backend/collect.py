from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from pathlib import Path

from .sources import LolchessSource, MetaTFTProSource, MetaTFTSource, OpggLiveSource, RiotHighEloSource, TacticsToolsSource
from .sources.base import TrainingSample, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "backend" / "data"


def current_patch() -> str:
    try:
        payload = json.loads((ROOT / "src" / "data" / "set18.generated.json").read_text("utf-8"))
        return str(payload.get("patch") or "18.1")
    except Exception:
        return "18.1"


def add_pro_aggregates(rows: list[TrainingSample], patch: str) -> list[TrainingSample]:
    try:
        snapshot = json.loads((DATA / "metatft.snapshot.json").read_text("utf-8"))
    except Exception:
        return []
    clusters = list(snapshot.get("clusters") or [])
    cluster_reps: list[tuple[str, list[str]]] = []
    for cluster in clusters:
        best_units: list[str] = []
        best_count = -1
        for level_rows in (cluster.get("options") or {}).values():
            for option in level_rows or []:
                units = [str(value) for value in option.get("units") or [] if value]
                count = int(option.get("count") or 0)
                if len(units) >= 4 and count > best_count:
                    best_units = units
                    best_count = count
        if not best_units:
            best_units = [str(value) for value in cluster.get("centroidUnits") or [] if value]
        if len(best_units) >= 4:
            cluster_reps.append((str(cluster.get("id")), best_units))

    board_groups: dict[str, list[TrainingSample]] = {}
    item_groups: dict[tuple[str, tuple[str, ...]], list[TrainingSample]] = {}
    seen_board_observations: set[tuple] = set()
    seen_item_observations: set[tuple] = set()
    for row in rows:
        # Only actual tracked pro boards are promoted into supervised aggregates. Full-lobby
        # participants remain valuable for priors/frequency, but their single-game outcomes
        # are too confounded to become direct regression targets.
        if row.source != "metatft-pro-live":
            continue
        match_id = str(row.context_id or "").split(":", 1)[0]
        if row.sample_kind in {"pro_final_board", "high_elo_final_board"}:
            observation = (match_id, tuple(sorted(row.units)), round(row.avg_placement, 4))
            if observation in seen_board_observations:
                continue
            seen_board_observations.add(observation)
            board = set(row.units)
            best_id = None
            best_similarity = 0.0
            for cluster_id, rep in cluster_reps:
                other = set(rep)
                similarity = len(board & other) / max(1, len(board | other))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_id = cluster_id
            if best_id and best_similarity >= 0.42:
                board_groups.setdefault(best_id, []).append(row)
        elif row.sample_kind in {"pro_item_holder_build", "high_elo_item_holder_build"} and row.units:
            observation = (match_id, row.units[0], tuple(sorted(row.items)), round(row.avg_placement, 4))
            if observation in seen_item_observations:
                continue
            seen_item_observations.add(observation)
            key = (row.units[0], tuple(sorted(row.items)))
            item_groups.setdefault(key, []).append(row)

    output: list[TrainingSample] = []
    rep_by_cluster = dict(cluster_reps)
    for cluster_id, group in board_groups.items():
        if len(group) < 3:
            continue
        games = len(group)
        avg = sum(row.avg_placement for row in group) / games
        top4 = sum(1.0 for row in group if row.avg_placement <= 4) / games
        wins = sum(1.0 for row in group if row.avg_placement == 1) / games
        regions = {row.region for row in group}
        ages = [row.age_hours for row in group if row.age_hours is not None]
        output.append(TrainingSample(
            source="metatft-pro-aggregate",
            patch=patch,
            region=next(iter(regions)) if len(regions) == 1 else "MULTI",
            units=rep_by_cluster[cluster_id],
            level=len(rep_by_cluster[cluster_id]),
            avg_placement=avg,
            top4_rate=top4,
            win_rate=wins,
            games=games,
            source_weight=1.10,
            sample_kind="pro_cluster_aggregate",
            context_id=f"pro-cluster:{cluster_id}",
            age_hours=min(ages) if ages else None,
        ))
    for (holder, items), group in item_groups.items():
        if len(group) < 2:
            continue
        games = len(group)
        avg = sum(row.avg_placement for row in group) / games
        top4 = sum(1.0 for row in group if row.avg_placement <= 4) / games
        wins = sum(1.0 for row in group if row.avg_placement == 1) / games
        ages = [row.age_hours for row in group if row.age_hours is not None]
        output.append(TrainingSample(
            source="metatft-pro-aggregate",
            patch=patch,
            region="MULTI",
            units=[holder],
            items=list(items),
            item_holders={holder: list(items)},
            level=8,
            avg_placement=avg,
            top4_rate=top4,
            win_rate=wins,
            games=games,
            source_weight=1.08,
            sample_kind="pro_item_holder_aggregate",
            context_id=f"pro-item:{holder}:{'|'.join(items)}",
            age_hours=min(ages) if ages else None,
        ))
    return output


async def main() -> None:
    sources = [
        RiotHighEloSource(),
        MetaTFTProSource(),
        MetaTFTSource(),
        OpggLiveSource(),
        LolchessSource(),
        TacticsToolsSource(),
    ]
    all_rows: list[TrainingSample] = []
    health: dict[str, dict] = {}
    for source in sources:
        try:
            rows = await source.collect()
            health[source.name] = {"ok": True, "samples": len(rows)}
            if getattr(source, "region_counts", None):
                health[source.name]["regions"] = source.region_counts
            if getattr(source, "diagnostics", None):
                health[source.name]["diagnostics"] = source.diagnostics
            all_rows.extend(rows)
            snapshot = getattr(source, "snapshot", None)
            if snapshot:
                DATA.mkdir(parents=True, exist_ok=True)
                (DATA / f"{source.name}.snapshot.json").write_text(
                    json.dumps(snapshot, ensure_ascii=False, indent=2),
                    "utf-8",
                )
            print(f"{source.name}: {len(rows):,} samples")
        except Exception as exc:
            health[source.name] = {"ok": False, "samples": 0, "error": str(exc)}
            print(f"{source.name}: FAILED: {exc}")

    live_patch = current_patch()
    all_rows = [
        row for row in all_rows
        if row.patch in {live_patch, "current-live"} or row.patch.startswith(f"{live_patch}.")
    ]

    pro_aggregates = add_pro_aggregates(all_rows, live_patch)
    all_rows.extend(pro_aggregates)
    health["metatft-pro-aggregate"] = {"ok": True, "samples": len(pro_aggregates)}
    print(f"metatft-pro-aggregate: {len(pro_aggregates):,} samples")

    # Deduplicate exact aggregate rows from repeated source pages while preserving source
    # diversity. Riot individual matches naturally remain distinct because their labels vary.
    deduped: list[TrainingSample] = []
    seen: set[tuple] = set()
    for row in all_rows:
        key = (
            row.source,
            row.patch,
            row.region,
            tuple(sorted(row.units)),
            tuple(sorted(row.items)),
            row.level,
            round(row.avg_placement, 4),
            row.games,
            row.sample_kind,
            row.context_id,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    # Cross-source agreement is a reliability feature, not a way to duplicate rows. Exact
    # board/item signatures seen by independent sources receive a small boost; disagreement
    # in their supervised target reduces both samples' influence.
    consensus_groups: dict[tuple, list[TrainingSample]] = {}
    for row in deduped:
        # Cross-source reliability should compare aggregate observations, not a single lucky
        # or unlucky placement from one player. Individual pro/lobby rows are handled through
        # high-Elo priors below instead.
        if row.source in {"metatft-pro-live", "metatft-high-elo-lobby"} and row.games <= 1:
            continue
        family = "item" if "item_holder" in row.sample_kind else "board"
        signature = (
            family,
            tuple(sorted(row.units)),
            tuple(sorted(row.items)),
            int(row.level) if family == "board" else 0,
        )
        consensus_groups.setdefault(signature, []).append(row)
    for rows in consensus_groups.values():
        sources = {row.source for row in rows}
        if len(sources) < 2:
            continue
        targets = [row.target_strength for row in rows]
        mean = sum(targets) / len(targets)
        variance = sum((value - mean) ** 2 for value in targets) / len(targets)
        disagreement = min(0.45, math.sqrt(variance) * 2.4)
        agreement = 1.0 - disagreement
        for row in rows:
            row.consensus_sources = len(sources)
            row.agreement = agreement

    # OP.GG and MetaTFT rarely expose the exact same final-board variant. Treat a strong
    # fuzzy overlap as independent confirmation while capping the number of boosts per OP.GG
    # comp so one popular comp cannot inflate hundreds of nearby variants.
    opgg_boards = [
        row for row in deduped
        if row.source == "opgg-live" and row.sample_kind == "opgg_comp" and len(row.units) >= 4
    ]
    metatft_boards = [
        row for row in deduped
        if row.source == "metatft" and row.sample_kind == "exact_comp_option" and len(row.units) >= 4
    ]
    fuzzy_confirmed = 0
    for opgg_row in opgg_boards:
        target = set(opgg_row.units)
        candidates: list[tuple[float, TrainingSample]] = []
        for row in metatft_boards:
            if abs(len(row.units) - len(opgg_row.units)) > 2:
                continue
            board = set(row.units)
            similarity = len(target & board) / max(len(target), len(board))
            if similarity >= 0.68:
                candidates.append((similarity, row))
        candidates.sort(key=lambda value: value[0], reverse=True)
        for similarity, row in candidates[:12]:
            target_gap = abs(row.target_strength - opgg_row.target_strength)
            agreement = max(0.55, 1.0 - target_gap * 1.55)
            row.consensus_sources = max(row.consensus_sources, 2)
            row.agreement = min(row.agreement, agreement) if row.agreement < 1.0 else agreement
            # The OP.GG row remains useful as an external benchmark; don't need to boost it
            # once per similar MetaTFT variant.
            fuzzy_confirmed += 1

    write_jsonl(DATA / "training.jsonl", deduped)
    counts = Counter(row.source for row in deduped)
    DATA.mkdir(parents=True, exist_ok=True)
    consensus_rows = sum(1 for row in deduped if row.consensus_sources > 1)

    # High-Elo individual matches are excellent behavior/frequency signals but noisy labels
    # for absolute item strength. Persist compact priors for the online optimizer instead of
    # forcing every single-match item row into regression training.
    pro_unit: dict[str, dict[str, float]] = {}
    pro_item_holder: dict[str, dict[str, float]] = {}
    for row in deduped:
        if row.source not in {"metatft-pro-live", "metatft-high-elo-lobby"}:
            continue
        top4 = 1.0 if row.avg_placement <= 4 else 0.0
        win = 1.0 if row.avg_placement == 1 else 0.0
        if row.sample_kind in {"pro_final_board", "high_elo_final_board"}:
            for unit_id in set(row.units):
                stats = pro_unit.setdefault(unit_id, {"games": 0.0, "placementSum": 0.0, "top4": 0.0, "wins": 0.0})
                stats["games"] += 1.0
                stats["placementSum"] += row.avg_placement
                stats["top4"] += top4
                stats["wins"] += win
        elif row.sample_kind in {"pro_item_holder_build", "high_elo_item_holder_build"} and row.units:
            holder = row.units[0]
            for item_id in set(row.items):
                key = f"{item_id}::{holder}"
                stats = pro_item_holder.setdefault(key, {"games": 0.0, "placementSum": 0.0, "top4": 0.0, "wins": 0.0})
                stats["games"] += 1.0
                stats["placementSum"] += row.avg_placement
                stats["top4"] += top4
                stats["wins"] += win

    def finish_priors(values: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for key, stats in values.items():
            games = max(1.0, stats["games"])
            output[key] = {
                "games": int(games),
                "avgPlacement": stats["placementSum"] / games,
                "top4Rate": stats["top4"] / games,
                "winRate": stats["wins"] / games,
            }
        return output

    (DATA / "high-elo-priors.json").write_text(
        json.dumps({
            "patch": live_patch,
            "units": finish_priors(pro_unit),
            "itemHolders": finish_priors(pro_item_holder),
        }, ensure_ascii=False, indent=2),
        "utf-8",
    )
    (DATA / "source-health.json").write_text(
        json.dumps({
            "sources": health,
            "samples": dict(counts),
            "total": len(deduped),
            "crossSourceRows": consensus_rows,
            "fuzzyCrossSourceMatches": fuzzy_confirmed,
            "patch": live_patch,
            "highEloUnitPriors": len(pro_unit),
            "highEloItemHolderPriors": len(pro_item_holder),
        }, indent=2),
        "utf-8",
    )
    print(f"total: {len(deduped):,} -> backend/data/training.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
