from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.evaluation import evaluate_all_shared_runs, plot_training_history, select_best_run, write_ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate trained CNN runs on a chosen split.')
    parser.add_argument('--config', default='configs/cnn/base.json')
    parser.add_argument('--split', default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--plot-histories', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=True)
    results = evaluate_all_shared_runs(config, split_name=args.split)
    ranking_path = config.output.reports_dir / f'shared_ranking_{args.split}.csv'
    ranked = write_ranking(results, ranking_path)
    best_path = config.output.reports_dir / f'best_shared_{args.split}.json'
    best = select_best_run(ranked, best_path)
    if args.plot_histories:
        for row in ranked:
            run_dir = Path(row['run_dir'])
            history_path = run_dir / 'history.csv'
            if history_path.exists():
                plot_training_history(history_path, run_dir / 'training_curves.png')
    print(f'ranking={ranking_path}')
    print(f'best_run={best["run_id"]}')


if __name__ == '__main__':
    main()
