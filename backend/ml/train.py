from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from backend.app.catalog import load_catalog
from backend.app.features import encode, make_feature_space
from backend.ml.model import build_ranker


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "backend" / "data" / "training.jsonl"
MODEL_DIR = ROOT / "backend" / "models"


def load_rows() -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit("Missing backend/data/training.jsonl. Run `npm run ml:collect` first.")
    rows = []
    for line in DATA_PATH.read_text("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def canonical_signature(row: dict, index: int = 0) -> str:
    """Group equivalent observations across sources so they cannot leak across splits."""
    kind = str(row.get("sample_kind") or "")
    family = "item" if "item_holder" in kind else "board"
    units = ",".join(sorted(str(value) for value in row.get("units") or []))
    items = ",".join(sorted(str(value) for value in row.get("items") or [])) if family == "item" else ""
    level = int(row.get("level") or 8) if family == "board" else 0
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
    is_item_model = any("item_holder" in kind for kind in allowed_kinds)
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
        calibration_predictions.append(model.predict(cal_x, verbose=0).reshape(-1))
        test_predictions.append(model.predict(test_x, verbose=0).reshape(-1))
        if external_matrix is not None:
            external_predictions.append(model.predict(external_matrix[0], verbose=0).reshape(-1))
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
    (MODEL_DIR / "training-summary.json").write_text(
        json.dumps({"board": board, "item": item}, indent=2),
        "utf-8",
    )


if __name__ == "__main__":
    main()
