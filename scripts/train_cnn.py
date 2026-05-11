from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.experiments import get_cnn_experiment, iter_cnn_experiments
from src.cnn.keras_models import build_shared_cnn_model
from src.cnn.training import train_experiments, write_training_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN Keras experiments.")
    parser.add_argument("--config", default="configs/cnn/base.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=True)
    experiments = _select_experiments(config, args.run_id, args.limit)

    if args.dry_run:
        for experiment in experiments:
            model = build_shared_cnn_model(config, experiment)
            print(f"{experiment.run_id}: params={model.count_params()}")
        return

    results = train_experiments(
        config,
        experiments=experiments,
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
    )
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else config.output.reports_dir / "shared_training_summary.csv"
    )
    write_training_summary(results, summary_path)
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['run_id']}: "
            f"status={result['status']}, "
            f"val_macro_f1={metrics['macro_f1']:.4f}, "
            f"val_accuracy={metrics['sparse_categorical_accuracy']:.4f}"
        )
    print(f"summary={summary_path}")


def _select_experiments(config, run_ids, limit):
    if run_ids:
        experiments = tuple(get_cnn_experiment(config, run_id) for run_id in run_ids)
    else:
        experiments = iter_cnn_experiments(config)

    if limit is not None:
        experiments = experiments[:limit]

    return experiments


if __name__ == "__main__":
    main()
