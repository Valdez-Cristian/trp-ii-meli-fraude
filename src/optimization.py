"""Probabilidades out-of-fold y busqueda del umbral optimo por costo."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config

THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def oof_probabilities(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = config.N_FOLDS,
    n_jobs: int | None = None,
    seed: int = config.RANDOM_STATE,
) -> np.ndarray:
    """Probabilidades out-of-fold con k-fold estratificado.

    Cada fila se predice con un modelo que no la vio en el ajuste. El pipeline
    se clona en cada fold, asi que todo el preprocesamiento se reajusta.

    Args:
        pipeline: Pipeline o estimador sin ajustar.
        X: Features de train.
        y: Etiquetas de train.
        n_folds: Cantidad de folds.
        n_jobs: Paralelismo de `cross_val_predict`.
        seed: Semilla de la particion en folds.

    Returns:
        Array con la probabilidad de la clase positiva para cada fila.
    """
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(
        clone(pipeline), X, y, cv=cv, method="predict_proba", n_jobs=n_jobs
    )
    return proba[:, 1]


def threshold_table(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = THRESHOLD_GRID,
    cost_fp: float = config.COSTO_FP,
    cost_fn: float = config.COSTO_FN,
) -> pd.DataFrame:
    """Calcula FP, FN y costo para cada umbral de la grilla.

    Args:
        y_true: Etiquetas reales (0/1).
        proba: Probabilidades predichas.
        thresholds: Umbrales a evaluar.
        cost_fp: Costo unitario de un FP.
        cost_fn: Costo unitario de un FN.

    Returns:
        DataFrame con columnas `threshold`, `fp`, `fn` y `cost`.
    """
    y_true = np.asarray(y_true).astype(bool)
    proba = np.asarray(proba)
    pred = proba[:, None] >= np.asarray(thresholds)[None, :]
    fp = (pred & ~y_true[:, None]).sum(axis=0)
    fn = (~pred & y_true[:, None]).sum(axis=0)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "fp": fp,
            "fn": fn,
            "cost": fp * cost_fp + fn * cost_fn,
        }
    )


def best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_fp: float = config.COSTO_FP,
    cost_fn: float = config.COSTO_FN,
    fp_max: int | None = config.FP_MAX,
    thresholds: np.ndarray = THRESHOLD_GRID,
) -> float:
    """Umbral que minimiza el costo total, sujeto a FP <= fp_max.

    `fp_max` se compara con los FP del conjunto en que se calculan las
    probabilidades (aqui, el train completo), no con una escala mensual.

    Args:
        y_true: Etiquetas reales (0/1).
        proba: Probabilidades out-of-fold.
        cost_fp: Costo unitario de un FP.
        cost_fn: Costo unitario de un FN.
        fp_max: Maximo de FP permitidos. None = sin restriccion.
        thresholds: Umbrales candidatos.

    Returns:
        El umbral optimo. En empates de costo, el menor umbral.

    Raises:
        ValueError: Si ningun umbral de la grilla cumple `fp_max`.
    """
    table = threshold_table(y_true, proba, thresholds, cost_fp, cost_fn)
    if fp_max is not None:
        table = table[table["fp"] <= fp_max]
        if table.empty:
            raise ValueError(
                f"Ningun umbral de la grilla cumple FP <= {fp_max}; "
                "ampliar la grilla o relajar la restriccion."
            )
    return float(table.loc[table["cost"].idxmin(), "threshold"])
