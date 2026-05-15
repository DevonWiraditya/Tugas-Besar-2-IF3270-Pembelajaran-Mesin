from __future__ import annotations

import numpy as np

from src.captioning.scratch.dense import DenseLayer
from src.captioning.scratch.embedding import EmbeddingLayer
from src.captioning.scratch.lstm import LSTMCell
from src.captioning.scratch.rnn import SimpleRNNCell


class ScratchRNNDecoder:
    def __init__(
        self,
        projection: DenseLayer,
        embedding: EmbeddingLayer,
        recurrent: SimpleRNNCell,
        output: DenseLayer,
        start_id: int,
        end_id: int,
    ) -> None:
        self.projection = projection
        self.embedding = embedding
        self.recurrent = recurrent
        self.output = output
        self.start_id = start_id
        self.end_id = end_id
        self.hidden_size = recurrent.recurrent_kernel.shape[0]

    def generate(self, feature: np.ndarray, max_length: int) -> list[int]:
        return self.generate_batch(
            np.asarray(feature, dtype=np.float32).reshape(1, -1),
            max_length=max_length,
        )[0]

    def generate_batch(self, features: np.ndarray, max_length: int) -> list[list[int]]:
        feature_batch = np.asarray(features, dtype=np.float32)
        batch_size = feature_batch.shape[0]
        hidden = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        tokens = [[self.start_id] for _ in range(batch_size)]
        finished = np.zeros((batch_size,), dtype=bool)
        hidden = self.recurrent.step(self.projection.forward(feature_batch), hidden)
        for _ in range(max_length - 1):
            probabilities = self.output.forward(hidden)
            next_ids = np.argmax(probabilities, axis=-1)
            for index, next_id in enumerate(next_ids.tolist()):
                if finished[index]:
                    continue
                tokens[index].append(int(next_id))
                if int(next_id) == self.end_id:
                    finished[index] = True
            if finished.all():
                break
            embedded = self.embedding.forward(next_ids.astype(np.int64))
            hidden = self.recurrent.step(embedded, hidden)
        return tokens


class ScratchLSTMDecoder:
    def __init__(
        self,
        projection: DenseLayer,
        embedding: EmbeddingLayer,
        recurrent: LSTMCell,
        output: DenseLayer,
        start_id: int,
        end_id: int,
    ) -> None:
        self.projection = projection
        self.embedding = embedding
        self.recurrent = recurrent
        self.output = output
        self.start_id = start_id
        self.end_id = end_id
        self.hidden_size = recurrent.hidden_units

    def generate(self, feature: np.ndarray, max_length: int) -> list[int]:
        return self.generate_batch(
            np.asarray(feature, dtype=np.float32).reshape(1, -1),
            max_length=max_length,
        )[0]

    def generate_batch(self, features: np.ndarray, max_length: int) -> list[list[int]]:
        feature_batch = np.asarray(features, dtype=np.float32)
        batch_size = feature_batch.shape[0]
        hidden = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        cell = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        tokens = [[self.start_id] for _ in range(batch_size)]
        finished = np.zeros((batch_size,), dtype=bool)
        hidden, cell = self.recurrent.step(self.projection.forward(feature_batch), hidden, cell)
        for _ in range(max_length - 1):
            probabilities = self.output.forward(hidden)
            next_ids = np.argmax(probabilities, axis=-1)
            for index, next_id in enumerate(next_ids.tolist()):
                if finished[index]:
                    continue
                tokens[index].append(int(next_id))
                if int(next_id) == self.end_id:
                    finished[index] = True
            if finished.all():
                break
            embedded = self.embedding.forward(next_ids.astype(np.int64))
            hidden, cell = self.recurrent.step(embedded, hidden, cell)
        return tokens
