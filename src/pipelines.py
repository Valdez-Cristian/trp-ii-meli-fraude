"""Ensamblado de pipelines de preprocesamiento (no calcula nada por si solo).

Todo el estado aprendido (imputacion, escalado, codificacion, estadisticos por
grupo) vive dentro de transformadores con `fit`, de modo que ajustar el
pipeline con datos de train es suficiente para evitar fuga de informacion.

Entrada esperada: DataFrame de pandas con las columnas producidas por
`features.add_basic_features` y `features.drop_unused_columns`.
"""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

from src import config
from src.features import (
    MISSING_LABEL,
    FrequencyEncoder,
    GroupAmountFeatures,
    GroupBehaviorFeatures,
    RareCategoryGrouper,
    add_basic_features,
    drop_unused_columns,
)

LINEAR = "linear"
TREE = "tree"


def fill_missing_labels(X: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza nulos por `FALTANTE` y fuerza texto en todas las columnas.

    Args:
        X: DataFrame con columnas categoricas.

    Returns:
        DataFrame con todas las columnas como texto y sin nulos.
    """
    return X.fillna(MISSING_LABEL).astype(str)


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convierte a pandas el DataFrame de Polars que entra a scikit-learn.

    Args:
        df: DataFrame de Polars con las features basicas ya agregadas.

    Returns:
        DataFrame de pandas.
    """
    return df.to_pandas()


def split_xy(df: pl.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Agrega las features sin estado y separa X de y.

    Args:
        df: DataFrame de Polars crudo (train o test).

    Returns:
        Tupla (X como DataFrame de pandas, y como array 0/1).
    """
    pdf = to_pandas(drop_unused_columns(add_basic_features(df)))
    y = pdf[config.TARGET_COLUMN].to_numpy()
    return pdf.drop(columns=[config.TARGET_COLUMN]), y


def _categorical_encoder(kind: str) -> BaseEstimator:
    """Codificador de categoricas segun el tipo de modelo."""
    if kind == LINEAR:
        return OneHotEncoder(
            handle_unknown="ignore", min_frequency=config.MIN_CATEGORY_FREQUENCY
        )
    return OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        min_frequency=config.MIN_CATEGORY_FREQUENCY,
    )


def _drop_score(columns: list[str], use_score: bool) -> list[str]:
    """Quita `score` y sus derivadas si `use_score` es False."""
    if use_score:
        return columns
    excluded = {"score", *config.SCORE_DERIVED_FEATURES}
    return [c for c in columns if c not in excluded]


def build_preprocessor(
    kind: str, use_score: bool = True, behavior: bool = False
) -> ColumnTransformer:
    """Construye el preprocesador para modelos lineales o de arboles.

    Lineal: imputacion por mediana + `StandardScaler` en numericas y
    `OneHotEncoder` en categoricas. Arboles: imputacion por mediana sin
    escalado y `OrdinalEncoder` en categoricas (valores desconocidos = -1).
    En ambos casos `j` usa `TargetEncoder` (con cross-fitting interno) y se
    agregan las features con estado de `src.features`.

    Args:
        kind: `LINEAR` o `TREE`.
        use_score: Si es False excluye `score` y las features derivadas.
        behavior: Si es True agrega el perfil por `j` (`GroupBehaviorFeatures`)
            y el target encoding de `g`.

    Returns:
        `ColumnTransformer` sin ajustar.

    Raises:
        ValueError: Si `kind` no es `LINEAR` ni `TREE`.
    """
    if kind not in (LINEAR, TREE):
        raise ValueError(f"kind debe ser '{LINEAR}' o '{TREE}', no '{kind}'")

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if kind == LINEAR:
        numeric_steps.append(("scale", StandardScaler()))

    group_steps = [
        ("group", GroupAmountFeatures()),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if kind == LINEAR:
        group_steps.append(("scale", StandardScaler()))

    categorical = _drop_score(config.CATEGORICAL_FEATURES, use_score)

    transformers = [
        ("num", Pipeline(numeric_steps), _drop_score(config.NUMERIC_FEATURES, use_score)),
        ("group", Pipeline(group_steps), config.GROUP_AMOUNT_INPUTS),
        (
            "freq",
            Pipeline(
                [
                    ("freq", FrequencyEncoder(columns=config.FREQUENCY_FEATURES)),
                    *([("scale", StandardScaler())] if kind == LINEAR else []),
                ]
            ),
            list(config.FREQUENCY_FEATURES),
        ),
        (
            "rare",
            Pipeline(
                [
                    ("group", RareCategoryGrouper(column=config.RARE_GROUPED_FEATURE)),
                    ("encode", _categorical_encoder(kind)),
                ]
            ),
            [config.RARE_GROUPED_FEATURE],
        ),
        (
            "target",
            TargetEncoder(
                cv=StratifiedKFold(
                    config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
                )
            ),
            config.TARGET_ENCODED_FEATURES + (config.BEHAVIOR_TARGET_EXTRA if behavior else []),
        ),
    ]
    if behavior:
        behavior_steps = [
            ("behavior", GroupBehaviorFeatures(use_score=use_score)),
            ("impute", SimpleImputer(strategy="median")),
        ]
        if kind == LINEAR:
            behavior_steps.append(("scale", StandardScaler()))
        transformers.append(
            (
                "behavior",
                Pipeline(behavior_steps),
                ["j", "log_monto"] + (["score"] if use_score else []),
            )
        )
    if categorical:
        transformers.insert(
            1,
            (
                "cat",
                Pipeline(
                    [
                        ("fill", FunctionTransformer(fill_missing_labels, feature_names_out="one-to-one")),
                        ("encode", _categorical_encoder(kind)),
                    ]
                ),
                categorical,
            ),
        )

    return ColumnTransformer(transformers, remainder="drop")


def build_anomaly_preprocessor() -> ColumnTransformer:
    """Preprocesador no supervisado para deteccion de anomalias (Tarea 5).

    No usa `fraude` ni `score` (ni sus derivadas) y no incluye codificacion
    por target. Todo se escala y la salida es densa, como piden PCA y LOF.

    Returns:
        `ColumnTransformer` sin ajustar.
    """
    categorical = _drop_score(config.CATEGORICAL_FEATURES, use_score=False)
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                _drop_score(config.NUMERIC_FEATURES, use_score=False),
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("fill", FunctionTransformer(fill_missing_labels, feature_names_out="one-to-one")),
                        ("encode", _categorical_encoder(LINEAR)),
                    ]
                ),
                categorical,
            ),
            (
                "group",
                Pipeline(
                    [
                        ("group", GroupAmountFeatures()),
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                config.GROUP_AMOUNT_INPUTS,
            ),
            (
                "freq",
                Pipeline(
                    [
                        ("freq", FrequencyEncoder(columns=config.FREQUENCY_FEATURES)),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(config.FREQUENCY_FEATURES),
            ),
            (
                "rare",
                Pipeline(
                    [
                        ("group", RareCategoryGrouper(column=config.RARE_GROUPED_FEATURE)),
                        ("encode", _categorical_encoder(LINEAR)),
                    ]
                ),
                [config.RARE_GROUPED_FEATURE],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_pipeline(
    kind: str, estimator: BaseEstimator, use_score: bool = True, behavior: bool = False
) -> Pipeline:
    """Une el preprocesador con un estimador.

    Args:
        kind: `LINEAR` o `TREE`.
        estimator: Clasificador de scikit-learn (o compatible).
        use_score: Si es False excluye `score` y las features derivadas.
        behavior: Ver `build_preprocessor`.

    Returns:
        `Pipeline` sin ajustar con los pasos `prep` y `model`.
    """
    return Pipeline(
        [("prep", build_preprocessor(kind, use_score, behavior)), ("model", estimator)]
    )
