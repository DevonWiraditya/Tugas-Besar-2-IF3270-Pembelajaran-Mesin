from __future__ import annotations

import numpy as np

from src.captioning.scratch.dense import DenseLayer
from src.captioning.scratch.embedding import EmbeddingLayer
from src.captioning.scratch.lstm import LSTMCell
from src.captioning.scratch.rnn import SimpleRNNCell


class ScratchRNNDecoder:
    def __init__(self, projection: DenseLayer, embedding: EmbeddingLayer, recurrent: SimpleRNNCell, output: DenseLayer, start_id: int, end_id: int) -> None:
        self.projection = projection
        self.embedding = embedding
        self.recurrent = recurrent
        self.output = output
        self.start_id = start_id
        self.end_id = end_id
        self.hidden_size = recurrent.recurrent_kernel.shape[0]

    def generate(self, feature: np.ndarray, max_length: int) -> list[int]:
        h_t = np.zeros((self.hidden_size,), dtype=np.float32)
        tokens = [self.start_id]
        x_t = self.projection.forward(feature)
        h_t = self.recurrent.step(x_t, h_t)
        for _ in range(max_length - 1):
            next_id = int(np.argmax(self.output.forward(h_t)))
            tokens.append(next_id)
            if next_id == self.end_id:
                break
            h_t = self.recurrent.step(self.embedding.forward(np.asarray(next_id, dtype=np.int64)), h_t)
        return tokens


class ScratchLSTMDecoder:
    def __init__(self, projection: DenseLayer, embedding: EmbeddingLayer, recurrent: LSTMCell, output: DenseLayer, start_id: int, end_id: int) -> None:
        self.projection = projection
        self.embedding = embedding
        self.recurrent = recurrent
        self.output = output
        self.start_id = start_id
        self.end_id = end_id
        self.hidden_size = recurrent.hidden_units

    def generate(self, feature: np.ndarray, max_length: int) -> list[int]:
        h_t = np.zeros((self.hidden_size,), dtype=np.float32)
        c_t = np.zeros((self.hidden_size,), dtype=np.float32)
        tokens = [self.start_id]
        x_t = self.projection.forward(feature)
        h_t, c_t = self.recurrent.step(x_t, h_t, c_t)
        for _ in range(max_length - 1):
            next_id = int(np.argmax(self.output.forward(h_t)))
            tokens.append(next_id)
            if next_id == self.end_id:
                break
            x_t = self.embedding.forward(np.asarray(next_id, dtype=np.int64))
            h_t, c_t = self.recurrent.step(x_t, h_t, c_t)
        return tokens
