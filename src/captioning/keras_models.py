from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from src.captioning.config import CaptioningConfig


def build_caption_decoder(
    config: CaptioningConfig,
    recurrent_type: str,
    layer_count: int,
    hidden_size: int,
    vocab_size: int,
) -> tf.keras.Model:
    feature_inputs = tf.keras.layers.Input(
        shape=(config.encoder.feature_dim,),
        name="image_features",
    )
    caption_inputs = tf.keras.layers.Input(
        shape=(config.text.max_caption_length - 1,),
        name="caption_tokens",
    )
    projected = tf.keras.layers.Dense(
        config.text.embedding_dim,
        activation="relu",
        name="feature_projection",
    )(feature_inputs)
    projected = tf.keras.layers.Reshape(
        (1, config.text.embedding_dim),
        name="feature_timestep",
    )(projected)
    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=config.text.embedding_dim,
        mask_zero=False,
        name="token_embedding",
    )(caption_inputs)
    sequence = tf.keras.layers.Concatenate(axis=1, name="pre_inject_sequence")(
        [projected, embedding]
    )
    x = _apply_recurrent_stack(
        sequence,
        recurrent_type=recurrent_type,
        layer_count=layer_count,
        hidden_size=hidden_size,
    )
    x = tf.keras.layers.Lambda(
        lambda tensor: tensor[:, 1:, :],
        output_shape=(config.text.max_caption_length - 1, hidden_size),
        name="drop_feature_timestep",
    )(x)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(vocab_size, activation="softmax"),
        name="token_probabilities",
    )(x)
    return _compile_model(
        tf.keras.Model(
            inputs=[feature_inputs, caption_inputs],
            outputs=outputs,
            name=f"{recurrent_type}_preinject_decoder",
        ),
        config,
    )


def _apply_recurrent_stack(
    inputs,
    recurrent_type: str,
    layer_count: int,
    hidden_size: int,
):
    recurrent_layer = tf.keras.layers.SimpleRNN
    if recurrent_type == "lstm":
        recurrent_layer = tf.keras.layers.LSTM
    x = inputs
    for index in range(layer_count):
        x = recurrent_layer(
            hidden_size,
            return_sequences=True,
            name=f"{recurrent_type}_{index + 1}",
        )(x)
    return x


def _compile_model(model: tf.keras.Model, config: CaptioningConfig) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.training.learning_rate
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )
    return model
