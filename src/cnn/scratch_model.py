from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.cnn.scratch_layers import (
    Conv2DLayer,
    DenseLayer,
    FlattenLayer,
    GlobalPooling2DLayer,
    LocallyConnected2DLayer,
    Pooling2DLayer,
    ScratchLayer,
)


@dataclass(frozen=True)
class ScratchCNNModel:
    layers: tuple[ScratchLayer, ...]
    name: str = "scratch_cnn"

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = np.asarray(x, dtype=np.float32)
        single = output.ndim in {1, 3}
        for layer in self.layers:
            output = layer.forward(output)
        if single and output.shape[0] == 1:
            return output[0]
        return output


def build_scratch_model_from_keras(model: Any) -> ScratchCNNModel:
    layers: list[ScratchLayer] = []

    for layer in model.layers:
        class_name = layer.__class__.__name__

        if class_name == "InputLayer":
            continue
        if class_name == "Conv2D":
            layers.append(_conv2d_from_keras(layer))
            continue
        if class_name == "LocallyConnected2D":
            layers.append(_locally_connected2d_from_keras(layer))
            continue
        if class_name == "MaxPooling2D":
            layers.append(_pooling_from_keras(layer, "max"))
            continue
        if class_name == "AveragePooling2D":
            layers.append(_pooling_from_keras(layer, "average"))
            continue
        if class_name == "GlobalMaxPooling2D":
            layers.append(GlobalPooling2DLayer(mode="max", name=layer.name))
            continue
        if class_name == "GlobalAveragePooling2D":
            layers.append(GlobalPooling2DLayer(mode="average", name=layer.name))
            continue
        if class_name == "Flatten":
            layers.append(FlattenLayer(name=layer.name))
            continue
        if class_name == "Dense":
            layers.append(_dense_from_keras(layer))
            continue

        raise ValueError(f"Unsupported Keras layer for scratch forward: {class_name}")

    return ScratchCNNModel(layers=tuple(layers), name=f"{model.name}_scratch")


def _conv2d_from_keras(layer: Any) -> Conv2DLayer:
    weights = layer.get_weights()
    kernel = weights[0].astype(np.float32, copy=False)
    bias = (
        weights[1].astype(np.float32, copy=False)
        if len(weights) > 1
        else np.zeros(kernel.shape[-1], dtype=np.float32)
    )
    return Conv2DLayer(
        kernel=kernel,
        bias=bias,
        strides=_pair(layer.strides),
        padding=layer.padding,
        activation=_activation_name(layer.activation),
        name=layer.name,
    )


def _locally_connected2d_from_keras(layer: Any) -> LocallyConnected2DLayer:
    weights = layer.get_weights()
    kernel = weights[0].astype(np.float32, copy=False)
    bias = (
        weights[1].astype(np.float32, copy=False)
        if len(weights) > 1
        else np.zeros(kernel.shape[-1], dtype=np.float32)
    )
    return LocallyConnected2DLayer(
        kernel=kernel,
        bias=bias,
        kernel_size=_pair(layer.kernel_size),
        strides=_pair(layer.strides),
        padding=layer.padding,
        activation=_activation_name(layer.activation),
        name=layer.name,
    )


def _pooling_from_keras(layer: Any, mode: str) -> Pooling2DLayer:
    return Pooling2DLayer(
        pool_size=_pair(layer.pool_size),
        strides=_pair(layer.strides),
        padding=layer.padding,
        mode=mode,
        name=layer.name,
    )


def _dense_from_keras(layer: Any) -> DenseLayer:
    weights = layer.get_weights()
    kernel = weights[0].astype(np.float32, copy=False)
    bias = (
        weights[1].astype(np.float32, copy=False)
        if len(weights) > 1
        else np.zeros(kernel.shape[-1], dtype=np.float32)
    )
    return DenseLayer(
        kernel=kernel,
        bias=bias,
        activation=_activation_name(layer.activation),
        name=layer.name,
    )


def _activation_name(activation: Any) -> str:
    return getattr(activation, "__name__", None) or getattr(activation, "name", "linear")


def _pair(value: Any) -> tuple[int, int]:
    return int(value[0]), int(value[1])
