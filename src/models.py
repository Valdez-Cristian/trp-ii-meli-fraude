"""Catalogo de estimadores. Cada entrada es (tipo de preprocesador, fabrica)."""

import json
from typing import Callable

from sklearn.base import BaseEstimator
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.pipelines import LINEAR, TREE

MODELS: dict[str, tuple[str, Callable[[], BaseEstimator]]] = {
    "logistic": (
        LINEAR,
        lambda: LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
    ),
    # n_jobs=1 en los estimadores: el paralelismo se hace por fold en la CV.
    # Sin class_weight: se preservan las probabilidades para el umbral por costo.
    "random_forest": (
        TREE,
        lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=1,
            random_state=config.RANDOM_STATE,
        ),
    ),
    "lightgbm": (
        TREE,
        lambda: LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=1,
            random_state=config.RANDOM_STATE,
            verbose=-1,
        ),
    ),
}


def get_model(name: str) -> tuple[str, BaseEstimator]:
    """Devuelve el tipo de preprocesador y un estimador nuevo.

    Args:
        name: Clave de `MODELS`. Con sufijo `_tuned` aplica los hiperparametros
            guardados por `scripts/tune_models.py`.

    Returns:
        Tupla (kind, estimador sin ajustar).

    Raises:
        KeyError: Si el modelo no esta registrado.
    """
    base = name.removesuffix("_tuned")
    if base not in MODELS:
        raise KeyError(f"Modelo '{name}' no registrado. Disponibles: {list(MODELS)}")
    kind, factory = MODELS[base]
    estimator = factory()
    if name.endswith("_tuned"):
        path = config.OOF_DIR / f"{base}_best_params.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No hay hiperparametros afinados para '{base}'. "
                "Correr scripts/tune_models.py primero."
            )
        estimator.set_params(**json.loads(path.read_text()))
    return kind, estimator
