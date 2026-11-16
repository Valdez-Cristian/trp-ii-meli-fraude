"""Combinacion de modelos a partir de sus probabilidades out-of-fold."""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src import config
from src.calibration import logit


def oof_path(name: str):
    """Ruta del parquet con las OOF de un modelo."""
    return config.OOF_DIR / f"{name}.parquet"


def load_oof(names: list[str]) -> pd.DataFrame:
    """Une las OOF de varios modelos por `id`.

    Args:
        names: Nombres de modelos con OOF guardadas por `train_models.py`.

    Returns:
        DataFrame indexado por `id` con una columna de probabilidad por modelo.
    """
    merged = None
    for name in names:
        part = pl.read_parquet(oof_path(name)).rename({"proba": name})
        merged = part if merged is None else merged.join(part, on=config.ID_COLUMN)
    return merged.to_pandas().set_index(config.ID_COLUMN)


def mean_blend(P: pd.DataFrame) -> np.ndarray:
    """Promedio simple de las probabilidades de los modelos."""
    return P.mean(axis=1).to_numpy()


def stacked_logistic(
    P: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = config.N_FOLDS,
    seed: int = config.RANDOM_STATE + 1,
) -> np.ndarray:
    """Metamodelo logistico sobre el logit de las probabilidades base.

    Se predice con ajuste cruzado: cada fila recibe la salida de un metamodelo
    que no la vio. La semilla de los folds difiere de la de los modelos base.

    Args:
        P: Probabilidades out-of-fold, una columna por modelo.
        y: Etiquetas (0/1).
        n_folds: Folds del ajuste cruzado.
        seed: Semilla de los folds del metamodelo.

    Returns:
        Probabilidades del apilado, alineadas con las filas de `P`.
    """
    Z = logit(P.to_numpy())
    y = np.asarray(y)
    out = np.empty(len(y))
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for fit_idx, held_idx in cv.split(Z, y):
        meta = LogisticRegression(C=1.0, max_iter=1000).fit(Z[fit_idx], y[fit_idx])
        out[held_idx] = meta.predict_proba(Z[held_idx])[:, 1]
    return out


def meta_weights(P: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Coeficientes del metamodelo ajustado con todas las OOF (solo para reportar)."""
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(logit(P.to_numpy()), np.asarray(y))
    return pd.Series(meta.coef_[0], index=P.columns)
