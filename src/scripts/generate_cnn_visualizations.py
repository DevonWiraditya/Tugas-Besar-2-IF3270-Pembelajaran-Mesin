from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.data import records_to_arrays
from src.cnn.visualization import (
    convolution_layer_names,
    feature_activation_model,
    grad_cam_model,
    load_trained_shared_model,
    output_stem,
    save_feature_map_grid,
    save_grad_cam,
    select_records_for_visualization,
)
from src.utils.io import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CNN feature map and Grad-CAM visualizations.")
    parser.add_argument("--config", default="configs/cnn/base.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--summary-path", default="reports/cnn/shared_training_summary.csv")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--layer-name", default=None)
    parser.add_argument("--max-feature-maps", type=int, default=16)
    parser.add_argument("--output-dir", default="reports/cnn/visualizations")
    parser.add_argument("--summary-output-path", default="reports/cnn/visualization_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=True)
    model, run_id = load_trained_shared_model(config, args.run_id, args.summary_path)
    layer_names = convolution_layer_names(model)
    if not layer_names:
        raise ValueError("Model does not contain Conv2D layers.")

    layer_name = args.layer_name or layer_names[-1]
    activation_model = feature_activation_model(model, layer_name)
    cam_model = grad_cam_model(model, layer_name)
    records = select_records_for_visualization(config, args.split, args.max_images)
    output_dir = Path(args.output_dir)
    rows = []

    for record in records:
        x, _ = records_to_arrays([record], config)
        stem = output_stem(record)
        feature_path = output_dir / f"{stem}_{layer_name}_feature_maps.png"
        grad_cam_path = output_dir / f"{stem}_{layer_name}_grad_cam.png"
        save_feature_map_grid(
            model,
            x,
            layer_name=layer_name,
            output_path=feature_path,
            max_maps=args.max_feature_maps,
            activation_model=activation_model,
        )
        predicted_index, confidence = save_grad_cam(
            model,
            x,
            layer_name=layer_name,
            output_path=grad_cam_path,
            grad_model=cam_model,
        )
        rows.append(
            [
                run_id,
                args.split,
                portable_path(record.path),
                record.class_name,
                layer_name,
                config.data.class_names[predicted_index],
                config.data.class_names[predicted_index] == record.class_name,
                confidence,
                feature_path.as_posix(),
                grad_cam_path.as_posix(),
            ]
        )

    write_csv(
        args.summary_output_path,
        [
            "run_id",
            "split",
            "image_path",
            "true_class",
            "layer_name",
            "predicted_class",
            "is_correct",
            "confidence",
            "feature_maps_path",
            "grad_cam_path",
        ],
        rows,
    )
    print(f"run_id={run_id}")
    print(f"layer_name={layer_name}")
    print(f"images={len(rows)}")
    print(f"output_dir={output_dir}")
    print(f"summary_output_path={args.summary_output_path}")


def portable_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return value.as_posix()


if __name__ == "__main__":
    main()
