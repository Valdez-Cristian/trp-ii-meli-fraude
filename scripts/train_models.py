"""Calcula probabilidades out-of-fold y el umbral optimo de un modelo.

Usa solo train. Guarda las OOF y el umbral en data/processed/oof/.
Uso: python scripts/train_models.py [nombre_modelo] [--no-score]
"""

import argparse
import json
import logging

import polars as pl

from src import config
from src.data import load_train
from src.metrics import evaluate
from src.models import get_model
from src.optimization import best_threshold, oof_probabilities
from src.pipelines import build_pipeline, split_xy


def main() -> None:
    """Corre CV, elige el umbral sobre las OOF y persiste los resultados."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="logistic")
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()

    X, y = split_xy(load_train())
    kind, estimator = get_model(args.model)
    pipeline = build_pipeline(kind, estimator, use_score=not args.no_score)

    proba = oof_probabilities(pipeline, X, y, n_jobs=-1)
    threshold = best_threshold(y, proba)
    result = evaluate(y, proba, threshold)

    name = args.model + ("_sin_score" if args.no_score else "")
    config.OOF_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({config.ID_COLUMN: X[config.ID_COLUMN].to_numpy(), "proba": proba}).write_parquet(
        config.OOF_DIR / f"{name}.parquet"
    )
    (config.OOF_DIR / f"{name}_threshold.json").write_text(
        json.dumps({"threshold": threshold, "fp_max": config.FP_MAX}, indent=2)
    )

    print(f"Modelo: {name} (OOF sobre train, {config.N_FOLDS} folds)")
    for key, value in result.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
