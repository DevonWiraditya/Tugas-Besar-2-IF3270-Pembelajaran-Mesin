from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.evaluation import select_best_run_id
from src.cnn.experiments import get_cnn_experiment, iter_cnn_experiments
from src.cnn.keras_models import build_cnn_model
from src.cnn.training import train_experiments, write_training_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN Keras experiments.")
    parser.add_argument("--config", default="configs/cnn/base.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", action="append", default=None)
    parser.add_argument("--parameter-sharing", choices=("shared", "nonshared"), default="shared")
    parser.add_argument("--best-from-summary", action="store_true")
    parser.add_argument("--best-summary-path", default="reports/cnn/shared_training_summary.csv")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=True)
    _validate_selection(args)
    summary_path = _summary_path(config, args.summary_path, args.parameter_sharing)
    run_ids = _selected_run_ids(args)
    experiments = _select_experiments(config, run_ids, args.limit)

    if args.dry_run:
        for experiment in experiments:
            model = build_cnn_model(
                config,
                experiment,
                parameter_sharing=args.parameter_sharing,
            )
            print(f"{model.name}: params={model.count_params()}")
        return

    results = train_experiments(
        config,
        experiments=experiments,
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        parameter_sharing=args.parameter_sharing,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        verbose=args.verbose,
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


def _validate_selection(args) -> None:
    if (
        args.parameter_sharing == "nonshared"
        and args.run_id is None
        and not args.best_from_summary
        and args.limit is None
    ):
        raise SystemExit(
            "Non-shared training is expensive; pass --run-id, --best-from-summary, or --limit."
        )


def _summary_path(config, summary_path, parameter_sharing):
    if summary_path:
        return Path(summary_path)
    filename = (
        "shared_training_summary.csv"
        if parameter_sharing == "shared"
        else "nonshared_training_summary.csv"
    )
    return config.output.reports_dir / filename


def _selected_run_ids(args):
    run_ids = args.run_id
    if args.best_from_summary:
        run_ids = [select_best_run_id(Path(args.best_summary_path))]
    return run_ids


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
