from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from backend.app.catalog import load_catalog
from backend.app.features import encode, make_feature_space
from backend.ml.model import build_ranker
from backend.ml.position import CELL_COUNT, build_position_model, cell_index, encode_position, make_position_space
from backend.ml.stars import STAR_CLASSES, build_star_model, encode_star, make_star_space


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "backend" / "data" / "training.jsonl"
SNAPSHOT_PATH = ROOT / "backend" / "data" / "metatft.snapshot.json"
MODEL_DIR = ROOT / "backend" / "models"
LOBBY_CACHE_DIR = ROOT / "backend" / "data" / "lobby-cache"


def load_rows() -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit("Missing backend/data/training.jsonl. Run `npm run ml:collect` first.")
    rows = []
    for line in DATA_PATH.read_text("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_lobby_star_rows() -> tuple[list[dict], list[dict]]:
    """Extract star-level and reroll supervision from cached Riot Match-V1 lobbies.

    Riot final-board payloads contain `units[].tier`, which is the actual 1★/2★/3★ state.
    We keep one star row per unit and one reroll row per participant. A reroll board is
    defined from observed behavior (at least one 1/2/3-cost unit actually reached 3★), not
    from a hand-authored list of comps/champions.
    """
    catalog = load_catalog()
    star_rows: list[dict] = []
    reroll_rows: list[dict] = []
    if not LOBBY_CACHE_DIR.is_dir():
        return star_rows, reroll_rows
    for path in sorted(LOBBY_CACHE_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text("utf-8"))
        except Exception:
            continue
        info = payload.get("info") or {}
        if int(info.get("tft_set_number") or 0) != 18:
            continue
        if int(info.get("queue_id") or info.get("queueId") or 0) != 1100:
            continue
        match_id = str((payload.get("metadata") or {}).get("match_id") or path.stem)
        for participant_index, participant in enumerate(info.get("participants") or []):
            placement = int(participant.get("placement") or 0)
            units = list(participant.get("units") or [])
            board = [
                str(unit.get("character_id"))
                for unit in units
                if str(unit.get("character_id") or "") in catalog.champion_by_id
            ]
            if placement < 1 or placement > 8 or len(board) < 4:
                continue
            level = int(participant.get("level") or len(board))
            context = f"{match_id}:{participant_index}:{placement}"
            low_cost_three_star = 0
            total_three_star = 0
            for unit in units:
                unit_id = str(unit.get("character_id") or "")
                champion = catalog.champion_by_id.get(unit_id)
                if champion is None:
                    continue
                stars = max(1, min(3, int(unit.get("tier") or 1)))
                if stars >= 3:
                    total_three_star += 1
                    if int(champion.get("cost") or 9) <= 3:
                        low_cost_three_star += 1
                star_rows.append({
                    "source": "riot-lobby-star",
                    "sample_kind": "unit_star",
                    "board": board,
                    "units": board,
                    "unit": unit_id,
                    "level": level,
                    "stars": stars,
                    "placement": placement,
                    "context_id": context,
                })
            # More than one 3★ low-cost unit is stronger evidence, but the target remains a
            # probability so a single observed 3★ carry is sufficient to teach reroll comps.
            reroll_target = 1.0 if low_cost_three_star > 0 else 0.0
            reroll_rows.append({
                "source": "riot-lobby-reroll",
                "sample_kind": "reroll_board",
                "units": board,
                "traits": [],
                "items": [],
                "level": level,
                "target_strength": reroll_target,
                "training_weight": 1.0 + min(0.65, low_cost_three_star * 0.25),
                "context_id": context,
                "low_cost_three_star": low_cost_three_star,
                "three_star_count": total_three_star,
            })
    # Future collectors persist the same signal into training.jsonl as `unit_stars`, so the
    # star/reroll pipeline keeps working even if raw lobby-cache files are rotated away.
    seen_star = {(str(row.get("context_id")), str(row.get("unit"))) for row in star_rows}
    seen_reroll = {str(row.get("context_id")) for row in reroll_rows}
    if DATA_PATH.exists():
        for line in DATA_PATH.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            unit_stars = row.get("unit_stars") or {}
            board = [str(value) for value in row.get("units") or [] if str(value) in catalog.champion_by_id]
            context = str(row.get("context_id") or "")
            if not unit_stars or not context or len(board) < 4:
                continue
            level = int(row.get("level") or len(board))
            for unit_id, raw_stars in unit_stars.items():
                unit_id = str(unit_id)
                key = (context, unit_id)
                if key in seen_star or unit_id not in catalog.champion_by_id:
                    continue
                seen_star.add(key)
                star_rows.append({
                    "source": str(row.get("source") or "training-jsonl-star"),
                    "sample_kind": "unit_star",
                    "board": board,
                    "units": board,
                    "unit": unit_id,
                    "level": level,
                    "stars": max(1, min(3, int(raw_stars or 1))),
                    "placement": float(row.get("avg_placement") or 4.5),
                    "context_id": context,
                })
            if context not in seen_reroll:
                low_cost_three_star = sum(
                    1
                    for unit_id, raw_stars in unit_stars.items()
                    if int(raw_stars or 1) >= 3
                    and int(catalog.champion_by_id.get(str(unit_id), {}).get("cost") or 9) <= 3
                )
                seen_reroll.add(context)
                reroll_rows.append({
                    "source": str(row.get("source") or "training-jsonl-reroll"),
                    "sample_kind": "reroll_board",
                    "units": board,
                    "traits": [],
                    "items": [],
                    "level": level,
                    "target_strength": 1.0 if low_cost_three_star > 0 else 0.0,
                    "training_weight": float(row.get("training_weight") or 1.0),
                    "context_id": context,
                    "low_cost_three_star": low_cost_three_star,
                })
    return star_rows, reroll_rows


def canonical_signature(row: dict, index: int = 0) -> str:
    """Group equivalent observations across sources so they cannot leak across splits."""
    kind = str(row.get("sample_kind") or "")
    family = "item" if ("item_holder" in kind or "item_affinity" in kind or "item_pair" in kind) else "board"
    units = ",".join(sorted(str(value) for value in row.get("units") or []))
    items = ",".join(sorted(str(value) for value in row.get("items") or [])) if family == "item" else ""
    level = int(row.get("level") or 8) if family == "board" else 0
    if family == "item" and items:
        return f"{family}::{level}::{units or '*'}::{items}"
    if not units:
        return f"empty::{index}"
    return f"{family}::{level}::{units}::{items}"


def grouped_split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Train/calibration/test split by canonical content, independent of source identity."""
    groups: dict[str, list[dict]] = {}
    for index, row in enumerate(rows):
        key = canonical_signature(row, index)
        groups.setdefault(key, []).append(row)
    keys = list(groups)
    random.shuffle(keys)
    train_target = max(1, int(len(rows) * 0.78))
    calibration_target = max(1, int(len(rows) * 0.10))
    train: list[dict] = []
    calibration: list[dict] = []
    test: list[dict] = []
    for key in keys:
        if len(train) < train_target:
            train.extend(groups[key])
        elif len(calibration) < calibration_target:
            calibration.extend(groups[key])
        else:
            test.extend(groups[key])
    if not calibration or not test:
        shuffled = list(train)
        random.shuffle(shuffled)
        cut = max(2, len(shuffled) // 8)
        test = shuffled[-cut:]
        calibration = shuffled[-2 * cut:-cut]
        train = shuffled[:-2 * cut]
    return train, calibration, test


def weighted_mae(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.abs(actual - predicted), weights=weights))


def fit_linear_calibration(predicted: np.ndarray, actual: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weight_sum = max(1e-8, float(weights.sum()))
    px = float(np.sum(predicted * weights) / weight_sum)
    py = float(np.sum(actual * weights) / weight_sum)
    variance = float(np.sum(weights * (predicted - px) ** 2) / weight_sum)
    covariance = float(np.sum(weights * (predicted - px) * (actual - py)) / weight_sum)
    slope = covariance / variance if variance > 1e-8 else 1.0
    slope = float(np.clip(slope, 0.55, 1.55))
    intercept = float(np.clip(py - slope * px, -0.25, 0.25))
    return slope, intercept


def ranking_accuracy(actual: np.ndarray, predicted: np.ndarray, seed: int = 1801) -> float:
    if len(actual) < 2:
        return 0.5
    rng = np.random.default_rng(seed)
    correct = 0
    total = 0
    for _ in range(min(8000, len(actual) * 12)):
        left, right = rng.integers(0, len(actual), size=2)
        if left == right or abs(float(actual[left] - actual[right])) < 0.025:
            continue
        correct += int((actual[left] > actual[right]) == (predicted[left] > predicted[right]))
        total += 1
    return correct / max(1, total)


def train_model(
    all_rows: list[dict],
    model_name: str,
    allowed_kinds: set[str],
    external_kinds: set[str] | None = None,
) -> dict:
    rows = [row for row in all_rows if row.get("sample_kind") in allowed_kinds]
    external_rows = [
        row for row in all_rows
        if external_kinds and row.get("sample_kind") in external_kinds
    ]
    is_item_model = any(
        "item_holder" in kind or "item_affinity" in kind or "item_pair" in kind
        for kind in allowed_kinds
    )
    if not is_item_model:
        rows = [row for row in rows if len(row.get("units") or []) >= 3]
    if len(rows) < 100:
        raise SystemExit(f"Need >=100 usable {model_name} samples, got {len(rows)}")

    catalog = load_catalog()
    space = make_feature_space(catalog)
    train_rows, calibration_rows, test_rows = grouped_split(rows)

    source_counts = Counter(str(row.get("source") or "unknown") for row in train_rows)

    def matrix(source_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.stack(
            [
                encode(
                    catalog,
                    space,
                    units=row.get("units") or [],
                    traits=row.get("traits") or [],
                    items=(row.get("items") or []) if is_item_model else [],
                    level=int(row.get("level") or 8),
                    sample_kind=str(row.get("sample_kind") or "final_board"),
                )
                for row in source_rows
            ]
        )
        y = np.asarray([float(row["target_strength"]) for row in source_rows], dtype=np.float32)
        # Reliability already lives in each row's source/evidence/freshness/consensus weight.
        # A second inverse-frequency reweighting over-amplifies tiny, noisy sources and was
        # empirically worse on the canonical holdout split.
        weights = np.asarray([float(row["training_weight"]) for row in source_rows], dtype=np.float32)
        return x, y, weights

    train_x, train_y, train_w = matrix(train_rows)
    cal_x, cal_y, cal_w = matrix(calibration_rows)
    test_x, test_y, test_w = matrix(test_rows)
    external_matrix = matrix(external_rows) if external_rows else None
    import tensorflow as tf
    ensemble_size = max(1, min(5, int(os.getenv("HEXCOACH_ENSEMBLE_SIZE", "3"))))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    member_paths: list[str] = []
    calibration_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    external_predictions: list[np.ndarray] = []
    for member in range(ensemble_size):
        tf.keras.utils.set_random_seed(1801 + member * 137)
        model = build_ranker(space.size)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=5e-5),
        ]
        model.fit(
            train_x,
            train_y,
            sample_weight=train_w,
            validation_data=(cal_x, cal_y, cal_w),
            epochs=80,
            batch_size=min(128, max(16, len(train_x) // 10)),
            callbacks=callbacks,
            verbose=0,
        )
        # Direct inference avoids rebuilding Keras predict functions for every freshly-created
        # ensemble member. This removes TensorFlow retracing overhead during training/eval.
        calibration_predictions.append(np.asarray(model(cal_x, training=False)).reshape(-1))
        test_predictions.append(np.asarray(model(test_x, training=False)).reshape(-1))
        if external_matrix is not None:
            external_predictions.append(np.asarray(model(external_matrix[0], training=False)).reshape(-1))
        model_path = MODEL_DIR / (f"{model_name}.keras" if member == 0 else f"{model_name}.{member}.keras")
        model.save(model_path)
        member_paths.append(str(model_path.relative_to(ROOT)))
        print(f"{model_name}: trained ensemble member {member + 1}/{ensemble_size}")
        del model
        tf.keras.backend.clear_session()

    raw_calibration = np.mean(np.stack(calibration_predictions), axis=0)
    raw_test = np.mean(np.stack(test_predictions), axis=0)
    ensemble_std = np.std(np.stack(test_predictions), axis=0)
    fitted_slope, fitted_intercept = fit_linear_calibration(raw_calibration, cal_y, cal_w)
    calibrated_cal = np.clip(raw_calibration * fitted_slope + fitted_intercept, 0.0, 1.0)
    raw_cal_mae = weighted_mae(cal_y, raw_calibration, cal_w)
    calibrated_cal_mae = weighted_mae(cal_y, calibrated_cal, cal_w)
    if calibrated_cal_mae < raw_cal_mae * 0.995:
        slope, intercept = fitted_slope, fitted_intercept
        calibration_enabled = True
    else:
        slope, intercept = 1.0, 0.0
        calibration_enabled = False
    calibrated_test = np.clip(raw_test * slope + intercept, 0.0, 1.0)
    raw_mae = weighted_mae(test_y, raw_test, test_w)
    calibrated_mae = weighted_mae(test_y, calibrated_test, test_w)
    baseline = np.full_like(test_y, float(np.average(train_y, weights=train_w)))
    baseline_mae = weighted_mae(test_y, baseline, test_w)
    per_source: dict[str, dict[str, float | int]] = {}
    for source in sorted({str(row.get("source") or "unknown") for row in test_rows}):
        indexes = np.asarray([index for index, row in enumerate(test_rows) if str(row.get("source") or "unknown") == source])
        if not len(indexes):
            continue
        per_source[source] = {
            "samples": int(len(indexes)),
            "mae": weighted_mae(test_y[indexes], calibrated_test[indexes], test_w[indexes]),
        }
    external_evaluation: dict[str, float | int] | None = None
    if external_matrix is not None and external_predictions:
        external_x, external_y, external_w = external_matrix
        _ = external_x
        external_raw = np.mean(np.stack(external_predictions), axis=0)
        external_calibrated = np.clip(external_raw * slope + intercept, 0.0, 1.0)
        external_baseline = np.full_like(external_y, float(np.average(train_y, weights=train_w)))
        external_evaluation = {
            "samples": len(external_rows),
            "mae": weighted_mae(external_y, external_calibrated, external_w),
            "baselineMAE": weighted_mae(external_y, external_baseline, external_w),
            "rankingAccuracy": ranking_accuracy(external_y, external_calibrated),
        }
    metadata = {
        "model": member_paths[0],
        "members": member_paths,
        "ensembleSize": ensemble_size,
        "feature_size": space.size,
        "samples": len(rows),
        "trainSamples": len(train_rows),
        "calibrationSamples": len(calibration_rows),
        "testSamples": len(test_rows),
        "sampleKinds": dict(Counter(str(row.get("sample_kind")) for row in rows)),
        "sources": sorted({row.get("source", "unknown") for row in rows}),
        "sourceBalance": {
            "strategy": "native_reliability_weights",
            "trainCounts": dict(source_counts),
        },
        "calibration": {
            "enabled": calibration_enabled,
            "slope": slope,
            "intercept": intercept,
            "rawMAE": raw_cal_mae,
            "calibratedMAE": calibrated_cal_mae,
        },
        "evaluation": {
            "rawMAE": raw_mae,
            "calibratedMAE": calibrated_mae,
            "baselineMAE": baseline_mae,
            "rankingAccuracy": ranking_accuracy(test_y, calibrated_test),
            "improvementVsBaseline": (baseline_mae - calibrated_mae) / max(1e-8, baseline_mae),
            "ensemblePredictionStd": float(np.mean(ensemble_std)),
            "perSource": per_source,
        },
        "externalEvaluation": external_evaluation,
    }
    (MODEL_DIR / f"{model_name}.meta.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def load_item_affinity_rows() -> list[dict]:
    """Build globally consistent single-item holder labels from live MetaTFT data.

    Runtime affinity only sees ``(holder, item)``. Older training code emitted one row per
    comp cluster, which meant the *same input vector* could receive many conflicting labels
    depending on the comp it came from. Aggregate every holder/item pair first so supervision
    matches the information available to the model at inference time.
    """
    snapshot_path = ROOT / "backend" / "data" / "metatft.snapshot.json"
    if not snapshot_path.exists():
        return []
    snapshot = json.loads(snapshot_path.read_text("utf-8"))
    if snapshot.get("queue") != "LIVE" or not str(snapshot.get("patch") or "").startswith("18."):
        return []
    catalog = load_catalog()
    aggregates: dict[tuple[str, str], dict[str, object]] = {}
    for cluster in snapshot.get("clusters") or []:
        cluster_id = str(cluster.get("id") or "")
        for item_row in cluster.get("itemStats") or []:
            item_id = str(item_row.get("item") or "")
            if item_id not in catalog.item_by_id:
                continue
            for holder in item_row.get("units") or []:
                unit_id = str(holder.get("unit") or "")
                count = int(holder.get("count") or 0)
                if unit_id not in catalog.champion_by_id or count <= 0:
                    continue
                place_change = float(holder.get("placeChange") or 0.0)
                item_pick = max(0.0, float(holder.get("itemPick") or 0.0))
                key = (unit_id, item_id)
                aggregate = aggregates.setdefault(key, {
                    "count": 0,
                    "place_change_sum": 0.0,
                    "item_pick_sum": 0.0,
                    "clusters": set(),
                })
                aggregate["count"] = int(aggregate["count"]) + count
                aggregate["place_change_sum"] = float(aggregate["place_change_sum"]) + place_change * count
                aggregate["item_pick_sum"] = float(aggregate["item_pick_sum"]) + item_pick * count
                clusters = aggregate["clusters"]
                assert isinstance(clusters, set)
                clusters.add(cluster_id)

    rows: list[dict] = []
    for (unit_id, item_id), aggregate in aggregates.items():
        count = int(aggregate["count"])
        cluster_count = len(aggregate["clusters"])
        # One tiny cluster is too contextual to become a global holder prior. A well sampled
        # single cluster is still useful, while sparse observations need repeated contexts.
        if count < 10 or (cluster_count < 2 and count < 35):
            continue
        place_change = float(aggregate["place_change_sum"]) / max(1, count)
        item_pick = float(aggregate["item_pick_sum"]) / max(1, count)
        target = float(np.clip(0.5 - place_change * 0.48, 0.04, 0.96))
        weight = (
            0.55
            + min(1.45, np.log1p(count) / 4.0)
            + min(0.30, item_pick * 0.75)
            + min(0.25, np.log1p(cluster_count) / 7.0)
        )
        rows.append({
            "source": "metatft-item-affinity",
            "sample_kind": "item_affinity",
            "units": [unit_id],
            "items": [item_id],
            "traits": [],
            "level": 8,
            "target_strength": target,
            "training_weight": float(weight),
            "context_id": f"global:{unit_id}:{item_id}",
            "games": count,
            "clusters": cluster_count,
        })
    return rows


def load_item_pair_affinity_rows() -> list[dict]:
    """Build holder/item-pair labels aggregated across all live comp clusters.

    The pair model, like the affinity model, does not receive a comp id at runtime. Pooling
    holder support before creating targets removes contradictory labels for identical input
    vectors and makes the learned signal line up with the runtime co-occurrence index.
    """
    if not SNAPSHOT_PATH.exists():
        return []
    snapshot = json.loads(SNAPSHOT_PATH.read_text("utf-8"))
    if snapshot.get("queue") != "LIVE" or not str(snapshot.get("patch") or "").startswith("18."):
        return []
    catalog = load_catalog()
    allowed_items = {
        str(item["id"])
        for item in catalog.items
        if item.get("category") in {"completed", "artifact", "radiant", "emblem"}
    }
    rows: list[dict] = []
    global_item_support: Counter[str] = Counter()
    global_pair_support: Counter[tuple[str, str]] = Counter()
    holder_item_support: dict[str, Counter[str]] = defaultdict(Counter)
    holder_pair_support: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    holder_clusters: dict[str, set[str]] = defaultdict(set)
    for cluster in snapshot.get("clusters") or []:
        cluster_id = str(cluster.get("id") or "")
        for build in cluster.get("builds") or []:
            unit_id = str(build.get("unit") or "")
            count = int(build.get("count") or 0)
            items = [str(value) for value in build.get("items") or [] if str(value) in allowed_items]
            if unit_id in catalog.champion_by_id and count >= 3 and len(set(items)) >= 2:
                unique_items = sorted(set(items))
                holder_clusters[unit_id].add(cluster_id)
                holder_item_support[unit_id].update({item_id: count for item_id in unique_items})
                holder_pair_support[unit_id].update({pair: count for pair in combinations(unique_items, 2)})
                global_item_support.update({item_id: count for item_id in unique_items})
                global_pair_support.update({pair: count for pair in combinations(unique_items, 2)})

    for unit_id, item_support in holder_item_support.items():
        pair_support = holder_pair_support[unit_id]
        # A slightly wider global pool is affordable after aggregation and prevents one noisy
        # cluster's top items from deciding which negative examples exist.
        pool = [item_id for item_id, support in item_support.most_common(18) if support >= 12]
        for left, right in combinations(sorted(pool), 2):
            left_support = int(item_support[left])
            right_support = int(item_support[right])
            support = min(left_support, right_support)
            if support < 12:
                continue
            cooccurrence = int(pair_support[(left, right)])
            cosine = cooccurrence / max(1.0, float(np.sqrt(left_support * right_support)))
            conditional = cooccurrence / max(1.0, float(support))
            target = float(np.clip(cosine * 0.60 + conditional * 0.40, 0.0, 1.0))
            weight = 0.55 + min(1.35, np.log1p(support) / 4.2)
            if cooccurrence > 0:
                weight += min(0.45, np.log1p(cooccurrence) / 12.0)
            weight += min(0.20, np.log1p(len(holder_clusters[unit_id])) / 8.0)
            rows.append({
                "source": "metatft-item-pair",
                "sample_kind": "item_pair_affinity",
                "units": [unit_id],
                "items": [left, right],
                "traits": [],
                "level": 8,
                "target_strength": target,
                "training_weight": float(weight),
                "context_id": f"global:{unit_id}",
                "pair_count": cooccurrence,
                "left_support": left_support,
                "right_support": right_support,
                "clusters": len(holder_clusters[unit_id]),
            })

    # A small holder-agnostic layer makes global anti-synergies learnable too. This captures
    # cases such as Wound/Sunder items that are each popular but never coexist in any holder
    # build, even when no single champion has enough support for both items independently.
    global_pool = [item_id for item_id, support in global_item_support.most_common(40) if support >= 40]
    for left, right in combinations(sorted(global_pool), 2):
        left_support = int(global_item_support[left])
        right_support = int(global_item_support[right])
        cooccurrence = int(global_pair_support[(left, right)])
        cosine = cooccurrence / max(1.0, float(np.sqrt(left_support * right_support)))
        conditional = cooccurrence / max(1.0, float(min(left_support, right_support)))
        target = float(np.clip(cosine * 0.60 + conditional * 0.40, 0.0, 1.0))
        support = min(left_support, right_support)
        rows.append({
            "source": "metatft-item-pair-global",
            "sample_kind": "item_pair_affinity",
            "units": [],
            "items": [left, right],
            "traits": [],
            "level": 8,
            "target_strength": target,
            "training_weight": float(0.65 + min(1.15, np.log1p(support) / 4.5)),
            "context_id": "global",
            "pair_count": cooccurrence,
            "left_support": left_support,
            "right_support": right_support,
        })
    return rows


def load_position_rows() -> list[dict]:
    if not SNAPSHOT_PATH.exists():
        raise SystemExit("Missing backend/data/metatft.snapshot.json. Run `npm run ml:collect` first.")
    snapshot = json.loads(SNAPSHOT_PATH.read_text("utf-8"))
    output: list[dict] = []
    for cluster in snapshot.get("clusters") or []:
        cluster_id = str(cluster.get("id") or "")
        centroid = [str(value) for value in cluster.get("centroidUnits") or [] if value]
        positioning = cluster.get("positioning") or {}
        for unit_id, positions in (positioning.get("units") or {}).items():
            counts = np.zeros(CELL_COUNT, dtype=np.float32)
            for position in positions or []:
                index = cell_index(str(position.get("cell") or ""))
                if index is not None:
                    counts[index] += max(0, int(position.get("count") or 0))
            games = int(counts.sum())
            if games < 20:
                continue
            distribution = counts / max(1.0, counts.sum())
            board = list(dict.fromkeys(centroid + [str(unit_id)]))
            output.append({
                "cluster": cluster_id,
                "board": board,
                "unit": str(unit_id),
                "distribution": distribution.tolist(),
                "games": games,
            })
    return output


def split_position_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["cluster"]), []).append(row)
    keys = list(groups)
    random.shuffle(keys)
    train_cut = max(1, int(len(keys) * 0.78))
    cal_cut = max(train_cut + 1, int(len(keys) * 0.88))
    train_keys = set(keys[:train_cut])
    cal_keys = set(keys[train_cut:cal_cut])
    test_keys = set(keys[cal_cut:])
    return (
        [row for row in rows if row["cluster"] in train_keys],
        [row for row in rows if row["cluster"] in cal_keys],
        [row for row in rows if row["cluster"] in test_keys],
    )


def position_metrics(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    actual_cell = np.argmax(actual, axis=1)
    predicted_cell = np.argmax(predicted, axis=1)
    top3 = np.argpartition(predicted, -3, axis=1)[:, -3:]
    exact = (actual_cell == predicted_cell).astype(np.float32)
    top3_hit = np.asarray([actual_cell[i] in top3[i] for i in range(len(actual_cell))], dtype=np.float32)
    actual_row = actual_cell // 7
    predicted_row = predicted_cell // 7
    row_hit = (actual_row == predicted_row).astype(np.float32)
    actual_col = actual_cell % 7
    predicted_col = predicted_cell % 7
    distance = np.abs(actual_row - predicted_row) + np.abs(actual_col - predicted_col)
    return {
        "top1CellAccuracy": float(np.average(exact, weights=weights)),
        "top3CellAccuracy": float(np.average(top3_hit, weights=weights)),
        "rowAccuracy": float(np.average(row_hit, weights=weights)),
        "meanGridDistance": float(np.average(distance, weights=weights)),
    }


def split_star_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    # Keep every champion from the same participant board in the same split. Otherwise the
    # model can see seven units from a final board during train and the eighth during test.
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("context_id") or "unknown"), []).append(row)
    keys = list(groups)
    random.shuffle(keys)
    train_cut = max(1, int(len(keys) * 0.78))
    cal_cut = max(train_cut + 1, int(len(keys) * 0.88))
    train_keys = set(keys[:train_cut])
    cal_keys = set(keys[train_cut:cal_cut])
    return (
        [row for row in rows if str(row.get("context_id")) in train_keys],
        [row for row in rows if str(row.get("context_id")) in cal_keys],
        [row for row in rows if str(row.get("context_id")) not in train_keys | cal_keys],
    )


def star_metrics(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    actual_class = np.argmax(actual, axis=1)
    predicted_class = np.argmax(predicted, axis=1)
    exact = (actual_class == predicted_class).astype(np.float32)
    distance = np.abs(actual_class - predicted_class).astype(np.float32)
    three_mask = actual_class == 2
    three_recall = 0.0
    if np.any(three_mask):
        three_recall = float(np.average((predicted_class[three_mask] == 2).astype(np.float32), weights=weights[three_mask]))
    predicted_three = predicted_class == 2
    three_precision = 0.0
    if np.any(predicted_three):
        three_precision = float(np.average((actual_class[predicted_three] == 2).astype(np.float32), weights=weights[predicted_three]))
    return {
        "accuracy": float(np.average(exact, weights=weights)),
        "meanStarDistance": float(np.average(distance, weights=weights)),
        "threeStarRecall": three_recall,
        "threeStarPrecision": three_precision,
    }


def train_star_ranker(rows: list[dict]) -> dict:
    if len(rows) < 300:
        raise SystemExit(f"Need >=300 unit-star samples, got {len(rows)}")
    catalog = load_catalog()
    space = make_star_space(catalog)
    train_rows, calibration_rows, test_rows = split_star_rows(rows)
    if not calibration_rows or not test_rows:
        raise SystemExit("Not enough independent lobby boards for star validation/test split")

    star_counts = Counter(int(row["stars"]) for row in train_rows)
    total = max(1, sum(star_counts.values()))
    class_weight = {
        stars: float(np.clip(np.sqrt(total / max(1.0, STAR_CLASSES * count)), 0.65, 2.2))
        for stars, count in star_counts.items()
    }

    def matrix(source_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.stack([
            encode_star(
                catalog,
                space,
                list(row["board"]),
                str(row["unit"]),
                int(row["level"]),
            )
            for row in source_rows
        ])
        y = np.zeros((len(source_rows), STAR_CLASSES), dtype=np.float32)
        for index, row in enumerate(source_rows):
            y[index, max(0, min(STAR_CLASSES - 1, int(row["stars"]) - 1))] = 1.0
        weights = np.asarray([
            class_weight.get(int(row["stars"]), 1.0)
            for row in source_rows
        ], dtype=np.float32)
        return x, y, weights

    train_x, train_y, train_w = matrix(train_rows)
    cal_x, cal_y, cal_w = matrix(calibration_rows)
    test_x, test_y, test_w = matrix(test_rows)

    import tensorflow as tf

    ensemble_size = max(1, min(5, int(os.getenv("HEXCOACH_ENSEMBLE_SIZE", "3"))))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    member_paths: list[str] = []
    test_predictions: list[np.ndarray] = []
    for member in range(ensemble_size):
        tf.keras.utils.set_random_seed(2801 + member * 163)
        model = build_star_model(space.size)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=4e-5),
        ]
        model.fit(
            train_x,
            train_y,
            sample_weight=train_w,
            validation_data=(cal_x, cal_y, cal_w),
            epochs=100,
            batch_size=min(128, max(16, len(train_x) // 10)),
            callbacks=callbacks,
            verbose=0,
        )
        test_predictions.append(np.asarray(model(test_x, training=False)))
        model_path = MODEL_DIR / ("star_ranker.keras" if member == 0 else f"star_ranker.{member}.keras")
        model.save(model_path)
        member_paths.append(str(model_path.relative_to(ROOT)))
        print(f"star_ranker: trained ensemble member {member + 1}/{ensemble_size}")
        del model
        tf.keras.backend.clear_session()

    predictions = np.mean(np.stack(test_predictions), axis=0)
    metadata = {
        "model": member_paths[0],
        "members": member_paths,
        "ensembleSize": ensemble_size,
        "feature_size": space.size,
        "samples": len(rows),
        "boards": len({str(row.get("context_id")) for row in rows}),
        "trainSamples": len(train_rows),
        "calibrationSamples": len(calibration_rows),
        "testSamples": len(test_rows),
        "classCounts": {str(key): int(value) for key, value in sorted(Counter(int(row["stars"]) for row in rows).items())},
        "evaluation": star_metrics(test_y, predictions, test_w),
    }
    # Persist empirical per-unit/cost priors beside the neural model. They are not labels
    # invented by the app: these are smoothed frequencies from the same Riot observations.
    # Runtime blends them with the model to prevent a reroll-heavy board from spuriously
    # turning an unrelated 4/5-cost unit into a recommended 3★ target.
    unit_counts: dict[str, Counter[int]] = defaultdict(Counter)
    cost_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for row in rows:
        unit_id = str(row["unit"])
        stars = int(row["stars"])
        unit_counts[unit_id][stars] += 1
        champion = catalog.champion_by_id.get(unit_id, {})
        cost_counts[int(champion.get("cost") or 1)][stars] += 1

    def distribution(counter: Counter[int]) -> list[float]:
        total_count = max(1, sum(counter.values()))
        return [float(counter.get(stars, 0) / total_count) for stars in range(1, STAR_CLASSES + 1)]

    prior_payload = {
        "samples": len(rows),
        "units": {
            unit_id: {"games": sum(counts.values()), "distribution": distribution(counts)}
            for unit_id, counts in unit_counts.items()
        },
        "costs": {
            str(cost): {"games": sum(counts.values()), "distribution": distribution(counts)}
            for cost, counts in cost_counts.items()
        },
    }
    (MODEL_DIR / "star-priors.json").write_text(json.dumps(prior_payload, indent=2), "utf-8")
    metadata["priorFile"] = "backend/models/star-priors.json"
    (MODEL_DIR / "star_ranker.meta.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def train_position_ranker() -> dict:
    rows = load_position_rows()
    if len(rows) < 100:
        raise SystemExit(f"Need >=100 positioning samples, got {len(rows)}")
    catalog = load_catalog()
    space = make_position_space(catalog)
    train_rows, calibration_rows, test_rows = split_position_rows(rows)
    if not calibration_rows or not test_rows:
        raise SystemExit("Not enough independent positioning clusters for validation/test split")

    def matrix(source_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.stack([
            encode_position(catalog, space, list(row["board"]), str(row["unit"]))
            for row in source_rows
        ])
        y = np.asarray([row["distribution"] for row in source_rows], dtype=np.float32)
        weights = np.asarray([
            0.45 + min(1.75, float(np.log1p(int(row["games"]))) / 4.5)
            for row in source_rows
        ], dtype=np.float32)
        return x, y, weights

    train_x, train_y, train_w = matrix(train_rows)
    cal_x, cal_y, cal_w = matrix(calibration_rows)
    test_x, test_y, test_w = matrix(test_rows)

    import tensorflow as tf

    ensemble_size = max(1, min(5, int(os.getenv("HEXCOACH_ENSEMBLE_SIZE", "3"))))
    member_paths: list[str] = []
    test_predictions: list[np.ndarray] = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for member in range(ensemble_size):
        tf.keras.utils.set_random_seed(2401 + member * 151)
        model = build_position_model(space.size)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=4e-5),
        ]
        model.fit(
            train_x,
            train_y,
            sample_weight=train_w,
            validation_data=(cal_x, cal_y, cal_w),
            epochs=100,
            batch_size=min(128, max(16, len(train_x) // 8)),
            callbacks=callbacks,
            verbose=0,
        )
        test_predictions.append(np.asarray(model(test_x, training=False)))
        model_path = MODEL_DIR / ("position_ranker.keras" if member == 0 else f"position_ranker.{member}.keras")
        model.save(model_path)
        member_paths.append(str(model_path.relative_to(ROOT)))
        print(f"position_ranker: trained ensemble member {member + 1}/{ensemble_size}")
        del model
        tf.keras.backend.clear_session()

    predictions = np.mean(np.stack(test_predictions), axis=0)
    entropy = -np.sum(predictions * np.log(np.clip(predictions, 1e-8, 1.0)), axis=1)
    metadata = {
        "model": member_paths[0],
        "members": member_paths,
        "ensembleSize": ensemble_size,
        "feature_size": space.size,
        "samples": len(rows),
        "clusters": len({row["cluster"] for row in rows}),
        "trainSamples": len(train_rows),
        "calibrationSamples": len(calibration_rows),
        "testSamples": len(test_rows),
        "evaluation": {
            **position_metrics(test_y, predictions, test_w),
            "meanPredictionEntropy": float(np.average(entropy, weights=test_w)),
        },
    }
    (MODEL_DIR / "position_ranker.meta.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    random.seed(1801)
    np.random.seed(1801)
    rows = load_rows()
    board = train_model(
        rows,
        "board_ranker",
        {"final_board", "exact_comp_option", "early_board", "pro_cluster_aggregate"},
        {"opgg_comp"},
    )
    item = train_model(
        rows,
        "item_ranker",
        {"item_holder_build", "pro_item_holder_aggregate"},
        {"opgg_item_holder"},
    )
    affinity_rows = load_item_affinity_rows()
    affinity = train_model(
        affinity_rows,
        "item_affinity_ranker",
        {"item_affinity"},
    )
    pair_rows = load_item_pair_affinity_rows()
    pair = train_model(
        pair_rows,
        "item_pair_ranker",
        {"item_pair_affinity"},
    )
    star_rows, reroll_rows = load_lobby_star_rows()
    star = train_star_ranker(star_rows)
    reroll = train_model(
        reroll_rows,
        "reroll_ranker",
        {"reroll_board"},
    )
    position = train_position_ranker()
    (MODEL_DIR / "training-summary.json").write_text(
        json.dumps(
            {
                "board": board,
                "item": item,
                "itemAffinity": affinity,
                "itemPair": pair,
                "star": star,
                "reroll": reroll,
                "position": position,
            },
            indent=2,
        ),
        "utf-8",
    )


if __name__ == "__main__":
    main()
