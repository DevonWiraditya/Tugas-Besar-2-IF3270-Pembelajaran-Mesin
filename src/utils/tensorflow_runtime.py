from __future__ import annotations

import os


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def configure_tensorflow_runtime(
    use_gpu: bool = True,
    memory_growth: bool = True,
    mixed_precision: bool = False,
) -> dict:
    import tensorflow as tf

    errors = []

    if not use_gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            errors.append(str(exc))

    physical_gpus = tf.config.list_physical_devices("GPU")

    if use_gpu and memory_growth:
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                errors.append(str(exc))

    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    logical_gpus = tf.config.list_logical_devices("GPU")
    policy = tf.keras.mixed_precision.global_policy().name

    return {
        "physical_gpus": [device.name for device in physical_gpus],
        "logical_gpus": [device.name for device in logical_gpus],
        "use_gpu": use_gpu,
        "memory_growth": memory_growth,
        "mixed_precision": mixed_precision,
        "mixed_precision_policy": policy,
        "errors": errors,
    }
