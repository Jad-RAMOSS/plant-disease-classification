from typing import Tuple

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam


def build_model(
    class_count: int,
    img_shape: Tuple[int, int, int],
    learning_rate: float = 0.001,
    freeze_base: bool = False,
) -> Sequential:
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(
        include_top=False, weights='imagenet', input_shape=img_shape, pooling='max'
    )
    base_model.trainable = not freeze_base
    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(
            256,
            kernel_regularizer=regularizers.l2(l=0.016),
            activity_regularizer=regularizers.l1(0.006),
            bias_regularizer=regularizers.l1(0.006),
            activation='relu',
        ),
        Dropout(rate=0.45, seed=123),
        Dense(class_count, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
