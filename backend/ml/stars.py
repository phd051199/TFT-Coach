from __future__ import annotations

from typing import Any

import numpy as np

from backend.app.catalog import Catalog
from backend.ml.position import PositionFeatureSpace, encode_position, make_position_space


STAR_CLASSES = 3


def make_star_space(catalog: Catalog) -> PositionFeatureSpace:
    # Star prediction needs both the whole board and an explicit target-unit one-hot.
    # PositionFeatureSpace already encodes exactly that without leaking the star label.
    return make_position_space(catalog)


def encode_star(
    catalog: Catalog,
    space: PositionFeatureSpace,
    board_units: list[str],
    target_unit: str,
    level: int,
) -> np.ndarray:
    return encode_position(catalog, space, board_units, target_unit, level=level)


def build_star_model(input_size: int) -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_size,), name="star_features")
    x = tf.keras.layers.LayerNormalization()(inputs)
    residual = tf.keras.layers.Dense(192, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.08)(residual)
    x = tf.keras.layers.Dense(192, activation="swish")(x)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Dense(96, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.06)(x)
    x = tf.keras.layers.Dense(48, activation="swish")(x)
    output = tf.keras.layers.Dense(STAR_CLASSES, activation="softmax", name="star_distribution")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="hexcoach_star_ranker")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=7e-4, weight_decay=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="star_accuracy")],
    )
    return model

