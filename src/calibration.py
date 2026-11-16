"""Calibracion de probabilidades (isotonica o sigmoide/Platt)."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from src import config

EPS = 1e-6
METHODS = ("isotonic", "sigmoid")


def logit(p: np.ndarray) -> np.ndarray:
    """Logit con recorte para evitar infinitos."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Mapea probabilidades crudas a probabilidades calibradas.

    Args:
        method: `isotonic` (monotona, no parametrica) o `sigmoid` (Platt:
            regresion logistica sobre el logit de la probabilidad).
    """

    def __init__(self, method: str = "isotonic"):
        if method not in METHODS:
            raise ValueError(f"method debe estar en {METHODS}, no '{method}'")
        self.method = method

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "Calibrator":
        """Aprende el mapeo con probabilidades out-of-fold y etiquetas."""
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model_.fit(proba, y)
        else:
            self.model_ = LogisticRegression(C=1e6, max_iter=1000)
            self.model_.fit(logit(proba)[:, None], y)
        return self

    def predict(self, proba: np.ndarray) -> np.ndarray:
        """Devuelve probabilidades calibradas."""
        if self.method == "isotonic":
            return self.model_.predict(proba)
        return self.model_.predict_proba(logit(proba)[:, None])[:, 1]


def cross_fitted_calibration(
    proba: np.ndarray,
    y: np.ndarray,
    method: str = "isotonic",
    n_folds: int = config.N_FOLDS,
    seed: int = config.RANDOM_STATE,
) -> np.ndarray:
    """Calibra cada fila con un calibrador que no la vio al ajustarse.

    Evita medir la calibracion (y elegir el umbral) con un calibrador
    ajustado sobre las mismas filas.

    Args:
        proba: Probabilidades out-of-fold.
        y: Etiquetas (0/1).
        method: `isotonic` o `sigmoid`.
        n_folds: Folds del ajuste cruzado.
        seed: Semilla de los folds.

    Returns:
        Probabilidades calibradas, alineadas con `proba`.
    """
    proba, y = np.asarray(proba), np.asarray(y)
    out = np.empty(len(proba))
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for fit_idx, held_idx in cv.split(proba, y):
        calibrator = Calibrator(method).fit(proba[fit_idx], y[fit_idx])
        out[held_idx] = calibrator.predict(proba[held_idx])
    return out


def calibration_scores(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Brier score y log-loss (menor = mejor calibrado), con probabilidades recortadas.

    Args:
        y: Etiquetas (0/1).
        proba: Probabilidades predichas.

    Returns:
        Diccionario con `brier` y `log_loss`.
    """
    p = np.clip(proba, EPS, 1 - EPS)
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p))}
