from __future__ import annotations

import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf

from src.captioning.config import CaptioningConfig


def build_caption_decoder(config: CaptioningConfig, recurrent_type: str, layer_count: int, hidden_size: int, vocab_size: int) -> tf.keras.Model:
    feature_inputs = tf.keras.layers.Input(shape=(config.encoder.feature_dim,), name='image_features')
    caption_inputs = tf.keras.layers.Input(shape=(config.text.max_caption_length - 1,), name='caption_tokens')
    projected = tf.keras.layers.Dense(config.encoder.projection_dim, activation='relu', name='feature_projection')(feature_inputs)
    projected = tf.keras.layers.Reshape((1, config.encoder.projection_dim), name='feature_timestep')(projected)
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=config.text.embedding_dim, mask_zero=True, name='token_embedding')(caption_inputs)
    sequence = tf.keras.layers.Concatenate(axis=1, name='pre_inject_sequence')([projected, embedding])
    recurrent_layer = tf.keras.layers.SimpleRNN if recurrent_type == 'rnn' else tf.keras.layers.LSTM
    x = sequence
    for index in range(layer_count):
        x = recurrent_layer(hidden_size, return_sequences=index < layer_count - 1, name=f'{recurrent_type}_{index + 1}')(x)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax', name='token_probabilities')(x)
    model = tf.keras.Model(inputs=[feature_inputs, caption_inputs], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate), loss=config.training.loss, metrics=['sparse_categorical_accuracy'])
    return model
