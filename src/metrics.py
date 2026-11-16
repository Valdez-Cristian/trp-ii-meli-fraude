"""Metricas de evaluacion y funcion de costo oficial."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)

from src import config


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Cuenta TN, FP, FN y TP.

    Args:
        y_true: Etiquetas reales (0/1).
        y_pred: Predicciones binarias (0/1).

    Returns:
        Diccionario con las claves `tn`, `fp`, `fn` y `tp`.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tn": int((~y_true & ~y_pred).sum()),
        "fp": int((~y_true & y_pred).sum()),
        "fn": int((y_true & ~y_pred).sum()),
        "tp": int((y_true & y_pred).sum()),
    }


def total_cost(
    fp: float,
    fn: float,
    cost_fp: float = config.COSTO_FP,
    cost_fn: float = config.COSTO_FN,
) -> float:
    """Costo total = FP * cost_fp + FN * cost_fn.

    Args:
        fp: Cantidad de falsos positivos.
        fn: Cantidad de falsos negativos.
        cost_fp: Costo unitario de un FP.
        cost_fn: Costo unitario de un FN.

    Returns:
        Costo total.
    """
    return fp * cost_fp + fn * cost_fn


def evaluate(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    cost_fp: float = config.COSTO_FP,
    cost_fn: float = config.COSTO_FN,
) -> dict[str, float]:
    """Calcula PR-AUC, precision, recall, FP, FN y costo para un umbral.

    Args:
        y_true: Etiquetas reales (0/1).
        proba: Probabilidades predichas de la clase positiva.
        threshold: Umbral de decision (se predice 1 si proba >= umbral).
        cost_fp: Costo unitario de un FP.
        cost_fn: Costo unitario de un FN.

    Returns:
        Diccionario con `threshold`, `pr_auc`, `precision`, `recall`, `fp`,
        `fn`, `tp`, `tn` y `cost`.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    counts = confusion_counts(y_true, pred)
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        **counts,
        "cost": float(total_cost(counts["fp"], counts["fn"], cost_fp, cost_fn)),
    }
