from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from src.cnn.config import CNNConfig
from src.cnn.data import ImageRecord, load_cnn_dataset, records_to_arrays
from src.cnn.evaluation import select_best_run_id
from src.cnn.experiments import get_cnn_experiment
from src.cnn.keras_models import build_shared_cnn_model


def load_trained_shared_model(
    config: CNNConfig,
    run_id: str | None = None,
    summary_path: str | Path | None = None,
) -> tuple[tf.keras.Model, str]:
    selected_run_id = run_id or select_best_run_id(
        Path(summary_path) if summary_path else config.output.reports_dir / "shared_training_summary.csv"
    )
    experiment = get_cnn_experiment(config, selected_run_id)
    model = build_shared_cnn_model(config, experiment)
    weights_path = config.output.models_dir / selected_run_id / "weights.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing trained weights: {weights_path}")
    model.load_weights(weights_path)
    return model, selected_run_id


def convolution_layer_names(model: tf.keras.Model) -> list[str]:
    return [layer.name for layer in model.layers if layer.__class__.__name__ == "Conv2D"]


def feature_activation_model(model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
    return tf.keras.Model(model.inputs, model.get_layer(layer_name).output)


def grad_cam_model(model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
    return tf.keras.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.layers[-2].output, model.output],
    )


def select_records_for_visualization(
    config: CNNConfig,
    split: str = "test",
    max_images: int | None = None,
) -> tuple[ImageRecord, ...]:
    dataset = load_cnn_dataset(config)
    records = getattr(dataset, split)
    selected: list[ImageRecord] = []
    seen = set()

    for record in records:
        if record.class_name in seen:
            continue
        selected.append(record)
        seen.add(record.class_name)
        if max_images is not None and len(selected) >= max_images:
            break

    if max_images is not None and len(selected) < max_images:
        selected.extend(record for record in records if record not in selected)
        selected = selected[:max_images]

    return tuple(selected)


def save_feature_map_grid(
    model: tf.keras.Model,
    x: np.ndarray,
    layer_name: str,
    output_path: str | Path,
    max_maps: int = 16,
    columns: int = 4,
    activation_model: tf.keras.Model | None = None,
) -> None:
    selected_model = activation_model or tf.keras.Model(model.inputs, model.get_layer(layer_name).output)
    activation = selected_model.predict(x, verbose=0)[0]
    scores = np.mean(np.abs(activation), axis=(0, 1))
    indexes = np.argsort(scores)[::-1][:max_maps]
    selected = activation[:, :, indexes]
    rows = int(np.ceil(len(indexes) / columns))

    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.2, rows * 2.2))
    axes_array = np.asarray(axes).reshape(rows, columns)
    for position, axis in enumerate(axes_array.ravel()):
        axis.axis("off")
        if position >= len(indexes):
            continue
        axis.imshow(_normalize(selected[:, :, position]), cmap="viridis")
        axis.set_title(f"map {int(indexes[position])}", fontsize=8)

    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def save_grad_cam(
    model: tf.keras.Model,
    x: np.ndarray,
    layer_name: str,
    output_path: str | Path,
    alpha: float = 0.42,
    grad_model: tf.keras.Model | None = None,
) -> tuple[int, float]:
    heatmap, class_index, confidence = grad_cam_heatmap(model, x, layer_name, grad_model=grad_model)
    image = _normalize(x[0])
    overlay = overlay_heatmap(image, heatmap, alpha=alpha)
    figure, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("image", fontsize=9)
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].axis("off")
    axes[1].set_title("grad-cam", fontsize=9)
    axes[2].imshow(overlay)
    axes[2].axis("off")
    axes[2].set_title("overlay", fontsize=9)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return class_index, confidence


def grad_cam_heatmap(
    model: tf.keras.Model,
    x: np.ndarray,
    layer_name: str,
    grad_model: tf.keras.Model | None = None,
) -> tuple[np.ndarray, int, float]:
    selected_model = grad_model or grad_cam_model(model, layer_name)
    batch = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_output, penultimate, predictions = selected_model(batch)
        class_index = int(tf.argmax(predictions[0]))
        kernel, bias = model.layers[-1].get_weights()
        logits = tf.matmul(penultimate, tf.convert_to_tensor(kernel, dtype=tf.float32))
        logits = logits + tf.convert_to_tensor(bias, dtype=tf.float32)
        target = logits[:, class_index]

    gradients = tape.gradient(target, conv_output)
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_output[0] * weights[0], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    resized = Image.fromarray(np.uint8(heatmap.numpy() * 255)).resize(
        (x.shape[2], x.shape[1]),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0, class_index, float(predictions[0, class_index])


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    colors = plt.get_cmap("jet")(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * colors, 0.0, 1.0)


def output_stem(record: ImageRecord) -> str:
    return _safe_name(f"{record.class_name}_{record.path.stem}")


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum <= minimum:
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
