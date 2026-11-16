"""Afina hiperparametros de un modelo con busqueda aleatoria sobre train.

Uso: python -m scripts.tune_models <modelo> [--trials N] [--no-score]
Guarda los mejores parametros y el ranking en data/processed/oof/.
"""

import argparse

import pandas as pd

from src import config
from src.data import load_train
from src.pipelines import split_xy
from src.tuning import SEARCH_SPACES, save_best, search


def main() -> None:
    """Corre la busqueda y persiste el mejor resultado."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(SEARCH_SPACES))
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()

    X, y = split_xy(load_train())
    ranking = search(args.model, X, y, args.trials, use_score=not args.no_score)
    best = save_best(args.model, ranking)

    ranking.assign(params=ranking["params"].astype(str)).to_csv(
        config.OOF_DIR / f"{args.model}_tuning_ranking.csv", index=False
    )
    pd.options.display.width = 200
    pd.options.display.max_colwidth = 120
    cols = ["pr_auc", "cost", "fp", "fn", "threshold", "params"]
    print("\nTop 5:")
    print(ranking[cols].head(5).to_string())
    print(f"\nMejores parametros guardados: {best}")


if __name__ == "__main__":
    main()
