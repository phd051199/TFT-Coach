from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.app.catalog import Catalog
from backend.app.features import FeatureSpace, encode, make_feature_space


BOARD_ROWS = 4
BOARD_COLS = 7
CELL_COUNT = BOARD_ROWS * BOARD_COLS


def cell_index(cell: str) -> int | None:
    if not cell.startswith("cell_"):
        return None
    try:
        value = int(cell.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return value - 1 if 1 <= value <= CELL_COUNT else None


def cell_name(index: int) -> str:
    return f"cell_{index + 1}"


def display_grid_position(index: int) -> tuple[int, int]:
    """Convert MetaTFT/Riot cell order into the player's displayed board coordinates.

    The source numbers cell_1..cell_7 on the player's back row and cell_22..cell_28
    on the frontline. Our canvas is drawn top-to-bottom (frontline to backline), so only
    the row axis needs to be mirrored. Keep the underlying model/output cell index in the
    source convention so existing position models remain compatible.
    """
    if not 0 <= index < CELL_COUNT:
        raise ValueError(f"cell index out of range: {index}")
    source_row, col = divmod(index, BOARD_COLS)
    return BOARD_ROWS - 1 - source_row, col


@dataclass(frozen=True)
class PositionFeatureSpace:
    board: FeatureSpace
    champion_ids: tuple[str, ...]
    champion_index: dict[str, int]
    dense_size: int = 7

    @property
    def size(self) -> int:
        return self.board.size + len(self.champion_ids) + self.dense_size


def make_position_space(catalog: Catalog) -> PositionFeatureSpace:
    champion_ids = tuple(str(row["id"]) for row in catalog.champions)
    return PositionFeatureSpace(
        board=make_feature_space(catalog),
        champion_ids=champion_ids,
        champion_index={value: index for index, value in enumerate(champion_ids)},
    )


def encode_position(
    catalog: Catalog,
    space: PositionFeatureSpace,
    board_units: list[str],
    target_unit: str,
    level: int | None = None,
) -> np.ndarray:
    board_vector = encode(
        catalog,
        space.board,
        units=board_units,
        level=level or max(3, min(10, len(board_units))),
        sample_kind="final_board",
    )
    output = np.zeros(space.size, dtype=np.float32)
    output[: space.board.size] = board_vector
    target_offset = space.board.size
    index = space.champion_index.get(target_unit)
    if index is not None:
        output[target_offset + index] = 1.0

    champion = catalog.champion_by_id.get(target_unit, {})
    role = str(champion.get("role") or "")
    stats: dict[str, Any] = champion.get("stats") or {}
    cost = int(champion.get("cost") or 1)
    attack_range = float(stats.get("range") or stats.get("attackRange") or 1)
    dense = np.asarray(
        [
            cost / 5.0,
            min(5.0, attack_range) / 5.0,
            1.0 if ("Tank" in role or "Fighter" in role) else 0.0,
            1.0 if role.startswith("AD") else 0.0,
            1.0 if role.startswith("AP") else 0.0,
            1.0 if any(token in role for token in ("Carry", "Caster", "Reaper")) else 0.0,
            len(board_units) / 10.0,
        ],
        dtype=np.float32,
    )
    dense_offset = target_offset + len(space.champion_ids)
    output[dense_offset : dense_offset + len(dense)] = dense
    return output


def build_position_model(input_size: int) -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_size,), name="position_features")
    x = tf.keras.layers.LayerNormalization()(inputs)
    residual = tf.keras.layers.Dense(192, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.08)(residual)
    x = tf.keras.layers.Dense(192, activation="swish")(x)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Dense(96, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.06)(x)
    x = tf.keras.layers.Dense(64, activation="swish")(x)
    output = tf.keras.layers.Dense(CELL_COUNT, activation="softmax", name="cell_distribution")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="hexcoach_position_ranker")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=7e-4, weight_decay=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="cell_accuracy")],
    )
    return model
