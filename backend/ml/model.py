from __future__ import annotations

from typing import Any


def build_ranker(input_size: int) -> Any:
    # Import TensorFlow lazily so data collection and API fallback can run even before the
    # heavy ML environment is installed.
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_size,), name="features")
    x = tf.keras.layers.LayerNormalization()(inputs)
    residual = tf.keras.layers.Dense(192, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.10)(residual)
    x = tf.keras.layers.Dense(192, activation="swish")(x)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Dense(96, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.08)(x)
    x = tf.keras.layers.Dense(48, activation="swish")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="strength")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="hexcoach_board_ranker")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=8e-4, weight_decay=1e-5),
        loss=tf.keras.losses.Huber(delta=0.10),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
