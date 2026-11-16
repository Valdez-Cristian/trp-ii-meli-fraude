"""Compara variantes de LightGBM con varias semillas de CV (solo train).

Mide si una mejora es mayor que el ruido de la particion en folds. Para cada
variante reporta media y desviacion de PR-AUC y costo, crudo y tras calibrar
(isotonica con ajuste cruzado). Variantes: features con/sin perfil de
comportamiento y distintos `scale_pos_weight`.

Uso: python -m scripts.compare_variants features | spw
"""

import argparse

import numpy as np
import pandas as pd

from src import config
from src.calibration import calibration_scores, cross_fitted_calibration
from src.data import load_train
from src.metrics import evaluate
from src.models import get_model
from src.optimization import best_threshold, oof_probabilities
from src.pipelines import build_pipeline, split_xy

SEEDS = (42, 43, 44)
SPW_VALUES = (1.0, 3.0, 10.0, 19.0)  # 19 = relacion negativos/positivos


def run_variant(X, y, behavior: bool, spw: float) -> pd.DataFrame:
    """Evalua una variante en todas las semillas y devuelve una fila por semilla."""
    rows = []
    for seed in SEEDS:
        kind, est = get_model("lightgbm_tuned")
        est.set_params(scale_pos_weight=spw)
        proba = oof_probabilities(
            build_pipeline(kind, est, behavior=behavior), X, y, n_jobs=-1, seed=seed
        )
        raw = evaluate(y, proba, best_threshold(y, proba))
        cal = cross_fitted_calibration(proba, y, "isotonic", seed=seed + 100)
        cal_res = evaluate(y, cal, best_threshold(y, cal))
        rows.append(
            {
                "pr_auc": raw["pr_auc"],
                "cost_raw": raw["cost"],
                "thr_raw": raw["threshold"],
                "cost_iso": cal_res["cost"],
                "logloss_raw": calibration_scores(y, proba)["log_loss"],
                "logloss_iso": calibration_scores(y, cal)["log_loss"],
            }
        )
        print(f"  seed {seed}: pr_auc={raw['pr_auc']:.4f} cost={raw['cost']:.0f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    """Corre las variantes pedidas e imprime media y desviacion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=["features", "spw"])
    args = parser.parse_args()

    X, y = split_xy(load_train())
    if args.study == "features":
        variants = {"sin perfil j": (False, 1.0), "con perfil j": (True, 1.0)}
    else:
        variants = {f"spw={v:g}": (False, v) for v in SPW_VALUES}

    summary = {}
    for name, (behavior, spw) in variants.items():
        print(name, flush=True)
        df = run_variant(X, y, behavior, spw)
        summary[name] = {
            **{f"{c}_mean": df[c].mean() for c in df},
            "pr_auc_std": df["pr_auc"].std(),
            "cost_raw_std": df["cost_raw"].std(),
        }
    pd.options.display.width = 220
    pd.options.display.float_format = "{:,.4f}".format
    cols = ["pr_auc_mean", "pr_auc_std", "cost_raw_mean", "cost_raw_std",
            "cost_iso_mean", "logloss_raw_mean", "logloss_iso_mean", "thr_raw_mean"]
    print(pd.DataFrame(summary).T[cols].to_string())


if __name__ == "__main__":
    main()
