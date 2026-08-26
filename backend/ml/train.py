from __future__ import annotations

import json
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


def grouped_split(rows: list[dict], fraction: float = 0.86) -> tuple[list[dict], list[dict]]:
    """Split by source/context rather than random rows to reduce near-duplicate leakage."""
    groups: dict[str, list[dict]] = {}
    for index, row in enumerate(rows):
        context = row.get("context_id") or f"row-{index}"
        key = f"{row.get('source','unknown')}::{context}"
        groups.setdefault(key, []).append(row)
    keys = list(groups)
    random.shuffle(keys)
    target = max(1, int(len(rows) * fraction))
    train: list[dict] = []
    validation: list[dict] = []
    for key in keys:
        if len(train) < target:
            train.extend(groups[key])
        else:
            validation.extend(groups[key])
    if not validation:
        cut = max(1, len(train) // 8)
        validation = train[-cut:]
        train = train[:-cut]
    return train, validation


def train_model(rows: list[dict], model_name: str, allowed_kinds: set[str]) -> dict:
    rows = [row for row in rows if row.get("sample_kind") in allowed_kinds]
    if "item_holder_build" not in allowed_kinds:
        rows = [row for row in rows if len(row.get("units") or []) >= 3]
    if len(rows) < 100:
        raise SystemExit(f"Need >=100 usable {model_name} samples, got {len(rows)}")

    catalog = load_catalog()
    space = make_feature_space(catalog)
    train_rows, val_rows = grouped_split(rows)

    def matrix(source_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.stack(
            [
                encode(
                    catalog,
                    space,
                    units=row.get("units") or [],
                    traits=row.get("traits") or [],
                    items=row.get("items") or [],
                    level=int(row.get("level") or 8),
                    sample_kind=str(row.get("sample_kind") or "final_board"),
                )
                for row in source_rows
            ]
        )
        y = np.asarray([float(row["target_strength"]) for row in source_rows], dtype=np.float32)
        weights = np.asarray([float(row["training_weight"]) for row in source_rows], dtype=np.float32)
        return x, y, weights

    train_x, train_y, train_w = matrix(train_rows)
    val_x, val_y, val_w = matrix(val_rows)
    model = build_ranker(space.size)
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=5e-5),
    ]
    model.fit(
        train_x,
        train_y,
        sample_weight=train_w,
        validation_data=(val_x, val_y, val_w),
        epochs=80,
        batch_size=min(128, max(16, len(train_x) // 10)),
        callbacks=callbacks,
        verbose=2,
    )
    metrics = model.evaluate(val_x, val_y, sample_weight=val_w, verbose=0, return_dict=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}.keras"
    model.save(model_path)
    metadata = {
        "model": str(model_path.relative_to(ROOT)),
        "feature_size": space.size,
        "samples": len(rows),
        "trainSamples": len(train_rows),
        "validationSamples": len(val_rows),
        "sampleKinds": dict(Counter(str(row.get("sample_kind")) for row in rows)),
        "sources": sorted({row.get("source", "unknown") for row in rows}),
        "validation": {key: float(value) for key, value in metrics.items()},
    }
    (MODEL_DIR / f"{model_name}.meta.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    random.seed(1801)
    np.random.seed(1801)
    rows = load_rows()
    board = train_model(rows, "board_ranker", {"final_board", "exact_comp_option", "early_board"})
    item = train_model(rows, "item_ranker", {"item_holder_build"})
    (MODEL_DIR / "training-summary.json").write_text(
        json.dumps({"board": board, "item": item}, indent=2),
        "utf-8",
    )


if __name__ == "__main__":
    main()
