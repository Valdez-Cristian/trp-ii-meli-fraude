"""Ingesta con Polars, depuracion de filas y particion train/test.

Este modulo es el unico que produce el split 75/25. El test solo debe leerse
con `load_test`, y unicamente desde la evaluacion final.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)


def load_raw(path: Path = config.RAW_DATA_PATH) -> pl.DataFrame:
    """Lee el CSV crudo y renombra a `id` la primera columna (sin nombre).

    Args:
        path: Ruta al CSV original.

    Returns:
        DataFrame de Polars con las 250.000 filas originales.
    """
    df = pl.read_csv(path, try_parse_dates=True, infer_schema_length=10_000)
    return df.rename({df.columns[0]: config.ID_COLUMN})


def drop_duplicate_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Elimina filas con `id` duplicado conservando la primera aparicion.

    No reordena el DataFrame. Las filas sinteticas son las de `id` duplicado.

    Args:
        df: DataFrame con la columna `id`.

    Returns:
        DataFrame sin ids duplicados, en el orden original.
    """
    return df.unique(subset=[config.ID_COLUMN], keep="first", maintain_order=True)


def load_clean() -> pl.DataFrame:
    """Lee el CSV y elimina los ids duplicados (filas sinteticas).

    Returns:
        DataFrame con las 150.000 filas validas en su orden original.
    """
    return drop_duplicate_ids(load_raw())


def temporal_split(
    df: pl.DataFrame, fractions: tuple[float, float, float] = config.TEMPORAL_FRACTIONS
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Parte el DataFrame en train / calibracion / test respetando el tiempo.

    Ordena por `fecha` (desempata por `id`) y corta en bloques consecutivos.
    Se usa en el modulo de anomalias, donde el orden cronologico importa.

    Args:
        df: DataFrame depurado.
        fractions: Proporciones de train, calibracion y test.

    Returns:
        Tupla (train, calibracion, test), de la mas antigua a la mas reciente.
    """
    ordered = df.sort([config.DATE_COLUMN, config.ID_COLUMN])
    n = ordered.height
    i1 = int(n * fractions[0])
    i2 = int(n * (fractions[0] + fractions[1]))
    return ordered[:i1], ordered[i1:i2], ordered[i2:]


def split_train_test(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parte el DataFrame en train y test, aleatoria y estratificada por `fraude`.

    Se conserva el orden original de las filas dentro de cada particion.

    Args:
        df: DataFrame depurado, antes de cualquier transformacion.

    Returns:
        Tupla (train, test) con proporcion 1 - TEST_SIZE / TEST_SIZE.
    """
    indices = np.arange(df.height)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.TEST_SIZE,
        stratify=df[config.TARGET_COLUMN].to_numpy(),
        random_state=config.SPLIT_RANDOM_STATE,
    )
    return df[np.sort(train_idx)], df[np.sort(test_idx)]


def prepare_splits() -> tuple[Path, Path]:
    """Ejecuta la ingesta completa y persiste train y test en parquet.

    Returns:
        Tupla (ruta_train, ruta_test).
    """
    df = drop_duplicate_ids(load_raw())
    if df.height != config.EXPECTED_ROWS:
        logger.warning(
            "Filas tras deduplicar: %d (esperadas: %d)", df.height, config.EXPECTED_ROWS
        )

    train, test = split_train_test(df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.write_parquet(config.TRAIN_PATH)
    test.write_parquet(config.TEST_PATH)
    return config.TRAIN_PATH, config.TEST_PATH


def load_train() -> pl.DataFrame:
    """Lee el conjunto de entrenamiento (75%).

    Returns:
        DataFrame de Polars con train.
    """
    return pl.read_parquet(config.TRAIN_PATH)


def load_test() -> pl.DataFrame:
    """Lee el conjunto de test (25%). Usar solo en la evaluacion final.

    Returns:
        DataFrame de Polars con test.
    """
    return pl.read_parquet(config.TEST_PATH)
