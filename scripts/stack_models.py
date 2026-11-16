"""Stacking y calibracion sobre las OOF guardadas (solo train).

Compara cada modelo base, el promedio simple y el metamodelo logistico, cada
uno crudo y calibrado (isotonica / sigmoide, con ajuste cruzado). Guarda las
OOF del apilado en data/processed/oof/stack.parquet.

Uso: python -m scripts.stack_models [modelo ...]
"""

import sys

import pandas as pd
import polars as pl

from src import config
from src.calibration import calibration_scores, cross_fitted_calibration
from src.data import load_train
from src.metrics import evaluate
from src.optimization import best_threshold
from src.stacking import load_oof, mean_blend, meta_weights, stacked_logistic

DEFAULT_MODELS = ["logistic", "random_forest", "lightgbm_tuned"]


def main() -> None:
    """Evalua bases, promedio y apilado, crudos y calibrados."""
    names = sys.argv[1:] or DEFAULT_MODELS
    P = load_oof(names)
    train = load_train().to_pandas().set_index(config.ID_COLUMN)
    y = train.loc[P.index, config.TARGET_COLUMN].to_numpy()

    candidates = {name: P[name].to_numpy() for name in names}
    candidates["promedio"] = mean_blend(P)
    candidates["stack_logistico"] = stacked_logistic(P, y)

    rows = []
    for name, proba in candidates.items():
        variants = {"crudo": proba}
        for method in ("isotonic", "sigmoid"):
            variants[method] = cross_fitted_calibration(proba, y, method)
        for calib, p in variants.items():
            result = evaluate(y, p, best_threshold(y, p))
            rows.append(
                {"modelo": name, "calibracion": calib, **calibration_scores(y, p),
                 **{k: result[k] for k in ("pr_auc", "threshold", "fp", "fn", "cost")}}
            )
    table = pd.DataFrame(rows)
    pd.options.display.width = 220
    pd.options.display.float_format = "{:,.4f}".format
    print(table.to_string(index=False))
    print("\nPesos del metamodelo (logit de cada modelo):")
    print(meta_weights(P, y).to_string())

    pl.DataFrame(
        {config.ID_COLUMN: P.index.to_numpy(), "proba": candidates["stack_logistico"]}
    ).write_parquet(config.OOF_DIR / "stack.parquet")
    table.to_csv(config.OOF_DIR / "stack_comparison.csv", index=False)


if __name__ == "__main__":
    main()
