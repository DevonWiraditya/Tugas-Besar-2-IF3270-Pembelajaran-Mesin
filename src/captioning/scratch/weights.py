from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.captioning.data import Vocabulary
from src.captioning.scratch.dense import DenseLayer
from src.captioning.scratch.embedding import EmbeddingLayer
from src.captioning.scratch.lstm import LSTMCell
from src.captioning.scratch.model import ScratchLSTMDecoder, ScratchRNNDecoder
from src.captioning.scratch.rnn import SimpleRNNCell


def build_scratch_decoder_from_keras(
    model: tf.keras.Model,
    recurrent_type: str,
    vocabulary: Vocabulary,
):
    projection = _dense_from_layer(model.get_layer("feature_projection"), activation="relu")
    embedding = EmbeddingLayer(model.get_layer("token_embedding").get_weights()[0])
    output = _dense_from_time_distributed(model.get_layer("token_probabilities"))

    recurrent_layer = model.get_layer(f"{recurrent_type}_1")
    if recurrent_type == "rnn":
        recurrent = SimpleRNNCell(*_recurrent_weights(recurrent_layer))
        return ScratchRNNDecoder(
            projection=projection,
            embedding=embedding,
            recurrent=recurrent,
            output=output,
            start_id=vocabulary.start_id,
            end_id=vocabulary.end_id,
        )

    recurrent = LSTMCell(*_recurrent_weights(recurrent_layer))
    return ScratchLSTMDecoder(
        projection=projection,
        embedding=embedding,
        recurrent=recurrent,
        output=output,
        start_id=vocabulary.start_id,
        end_id=vocabulary.end_id,
    )


def _dense_from_layer(layer, activation: str | None = None) -> DenseLayer:
    kernel, bias = layer.get_weights()
    return DenseLayer(kernel=kernel, bias=bias, activation=activation)


def _dense_from_time_distributed(layer) -> DenseLayer:
    dense_layer = layer.layer
    kernel, bias = dense_layer.get_weights()
    return DenseLayer(kernel=kernel, bias=bias, activation="softmax")


def _recurrent_weights(layer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = layer.get_weights()
    kernel = np.asarray(weights[0], dtype=np.float32)
    recurrent_kernel = np.asarray(weights[1], dtype=np.float32)
    bias = np.asarray(weights[2], dtype=np.float32)
    return kernel, recurrent_kernel, bias
