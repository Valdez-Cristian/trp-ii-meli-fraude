"""Busqueda aleatoria de hiperparametros sobre train (solo con OOF)."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from src import config
from src.metrics import evaluate
from src.models import get_model
from src.optimization import best_threshold, oof_probabilities
from src.pipelines import build_pipeline

SEARCH_SPACES: dict[str, dict] = {
    "lightgbm": {
        "n_estimators": [200, 400, 600, 800],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [20, 50, 100, 200],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "reg_lambda": [0.0, 1.0, 5.0, 20.0],
    },
    "random_forest": {
        "n_estimators": [200, 400],
        "min_samples_leaf": [5, 10, 20, 50, 100],
        "max_features": ["sqrt", 0.2, 0.4],
        "max_depth": [None, 12, 20],
    },
}


def params_path(name: str) -> Path:
    """Ruta del JSON con los mejores hiperparametros de un modelo."""
    return config.OOF_DIR / f"{name}_best_params.json"


def search(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    n_trials: int,
    use_score: bool = True,
) -> pd.DataFrame:
    """Evalua `n_trials` combinaciones al azar y devuelve el ranking.

    Cada combinacion se evalua con k-fold estratificado sobre train. Se ordena
    por PR-AUC OOF (no depende del umbral); el costo se reporta con el umbral
    optimo de esas mismas OOF.

    Args:
        name: Modelo de `models.MODELS` (debe estar en `SEARCH_SPACES`).
        X: Features de train.
        y: Etiquetas de train.
        n_trials: Cantidad de combinaciones a probar.
        use_score: Si es False excluye `score` y derivadas.

    Returns:
        DataFrame ordenado por `pr_auc` descendente, con una columna `params`.
    """
    rows = []
    sampler = ParameterSampler(
        SEARCH_SPACES[name], n_iter=n_trials, random_state=config.RANDOM_STATE
    )
    for i, params in enumerate(sampler, 1):
        start = time.time()
        kind, estimator = get_model(name)
        estimator.set_params(**params)
        proba = oof_probabilities(
            build_pipeline(kind, estimator, use_score), X, y, n_jobs=-1
        )
        result = evaluate(y, proba, best_threshold(y, proba))
        rows.append({**result, "params": params})
        print(
            f"[{name}] {i}/{n_trials} pr_auc={result['pr_auc']:.4f} "
            f"cost={result['cost']:.0f} ({time.time() - start:.0f}s)",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def save_best(name: str, ranking: pd.DataFrame) -> dict:
    """Guarda en JSON los hiperparametros del mejor intento.

    Args:
        name: Modelo afinado.
        ranking: Salida de `search`.

    Returns:
        Los hiperparametros guardados.
    """
    best = ranking.loc[0, "params"]
    config.OOF_DIR.mkdir(parents=True, exist_ok=True)
    params_path(name).write_text(json.dumps(best, indent=2))
    return best
