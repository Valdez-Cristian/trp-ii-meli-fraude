"""Feature engineering.

Dos tipos de piezas:

* Sin estado (`add_basic_features`): expresiones de Polars que solo usan la
  fila actual. Se pueden aplicar antes del split sin riesgo de fuga.
* Con estado (clases sklearn): aprenden estadisticos en `fit` (medias por
  grupo, frecuencias, etc.) y deben ajustarse unicamente con datos de train.
"""

import math

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from src import config

MISSING_LABEL = "FALTANTE"
OTHER_LABEL = "OTRO"


# ---------------------------------------------------------------------------
# Features sin estado (Polars)
# ---------------------------------------------------------------------------

def add_basic_features(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega las features que no aprenden nada de los datos.

    Incluye banderas de valores negativos, componentes temporales, logaritmos,
    features derivadas de `score` y features de texto del titulo `i`.
    No elimina columnas ni modifica las originales.

    Args:
        df: DataFrame con las columnas originales del dataset.

    Returns:
        DataFrame con las columnas originales mas las nuevas.
    """
    hora = pl.col(config.DATE_COLUMN).dt.hour()
    dow = pl.col(config.DATE_COLUMN).dt.weekday()  # lunes = 1, domingo = 7
    score = pl.col("score")
    titulo = pl.col("i").fill_null("")
    largo = titulo.str.len_chars()

    df = df.with_columns(
        # Banderas de negativos (los valores se recortan solo dentro de los log)
        (pl.col("f") < 0).fill_null(False).cast(pl.Int8).alias("f_neg"),
        (pl.col("q") < 0).fill_null(False).cast(pl.Int8).alias("q_neg"),
        # Temporales
        hora.alias("hora"),
        (2 * math.pi * hora / 24).sin().alias("hora_sin"),
        (2 * math.pi * hora / 24).cos().alias("hora_cos"),
        dow.alias("dow"),
        (2 * math.pi * (dow - 1) / 7).sin().alias("dow_sin"),
        (2 * math.pi * (dow - 1) / 7).cos().alias("dow_cos"),
        dow.is_in([6, 7]).cast(pl.Int8).alias("es_fin_de_semana"),
        pl.when(hora <= 5)
        .then(pl.lit("madrugada"))
        .when(hora <= 9)
        .then(pl.lit("manana"))
        .when(hora <= 21)
        .then(pl.lit("dia"))
        .otherwise(pl.lit("noche"))
        .alias("hora_bin"),
        (hora <= 5).cast(pl.Int8).alias("es_madrugada"),
        # Escala
        pl.col("monto").clip(lower_bound=0).log1p().alias("log_monto"),
        pl.col("f").clip(lower_bound=0).log1p().alias("log_f"),
        pl.col("c").clip(lower_bound=0).log1p().alias("log_c"),
        pl.col("m").clip(lower_bound=0).log1p().alias("log_m"),
        pl.col("l").clip(lower_bound=0).log1p().alias("log_l"),
        # Derivadas de score
        pl.when(score <= 10)
        .then(pl.lit("(-1, 10]"))
        .when(score <= 30)
        .then(pl.lit("(10, 30]"))
        .when(score <= 50)
        .then(pl.lit("(30, 50]"))
        .when(score <= 70)
        .then(pl.lit("(50, 70]"))
        .when(score <= 85)
        .then(pl.lit("(70, 85]"))
        .otherwise(pl.lit("(85, 100]"))
        .alias("score_bin"),
        # Texto
        largo.alias("len_titulo"),
        titulo.str.count_matches(r"\S+").alias("n_palabras"),
        titulo.str.count_matches(r"\d").alias("cant_digitos"),
        (titulo.str.count_matches(r"\p{Lu}") / largo.clip(lower_bound=1)).alias(
            "pct_mayusculas"
        ),
        titulo.str.count_matches(r"[^\p{L}\p{N}\s]").alias("cant_simbolos"),
    )
    # Depende de log_monto, que recien existe en el paso anterior.
    return df.with_columns(
        (pl.col("score") * pl.col("log_monto")).alias("score_x_logmonto")
    )


def drop_unused_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Elimina las columnas descartadas (`config.DROPPED_COLUMNS`).

    Args:
        df: DataFrame con las columnas originales.

    Returns:
        DataFrame sin las columnas descartadas. Conserva `id` y `fraude`.
    """
    return df.drop([c for c in config.DROPPED_COLUMNS if c in df.columns])


# ---------------------------------------------------------------------------
# Features con estado (sklearn). Entrada: DataFrame de pandas.
# ---------------------------------------------------------------------------

def _keys(series: pd.Series) -> pd.Series:
    """Convierte una columna a claves de texto, con los nulos como `FALTANTE`."""
    return series.fillna(MISSING_LABEL).astype(str)


class GroupAmountFeatures(TransformerMixin, BaseEstimator):
    """Compara el monto de cada transaccion con el de su grupo.

    Columnas de entrada requeridas: `monto`, `log_monto`, `hora`, `g`, `j`.
    Columnas de salida: `ratio_monto_j`, `ratio_monto_hora`, `ratio_monto_g`,
    `monto_rel_j` y `monto_pct_j`.

    Los grupos con menos de `min_group_size` filas en `fit`, y los grupos no
    vistos en `transform`, usan el estadistico global. Esto evita que un grupo
    con una sola fila produzca un ratio trivialmente igual a 1 en train.

    Args:
        min_group_size: Tamano minimo de grupo para usar su propio estadistico.
    """

    _RATIO_SPECS = (
        ("ratio_monto_j", "j"),
        ("ratio_monto_hora", "hora"),
        ("ratio_monto_g", "g"),
    )

    def __init__(self, min_group_size: int = 5):
        self.min_group_size = min_group_size

    def _group_stat(self, keys: pd.Series, values: np.ndarray, how: str) -> pd.Series:
        frame = pd.DataFrame({"k": keys.to_numpy(), "v": values})
        agg = frame.groupby("k")["v"].agg([how, "count"])
        return agg.loc[agg["count"] >= self.min_group_size, how]

    def fit(self, X: pd.DataFrame, y=None):
        """Aprende medias, medianas y distribuciones por grupo (solo con X)."""
        monto = X["monto"].to_numpy(dtype=float)
        log_monto = X["log_monto"].to_numpy(dtype=float)

        self.mean_maps_ = {
            name: self._group_stat(_keys(X[col]), monto, "mean")
            for name, col in self._RATIO_SPECS
        }
        self.global_mean_ = float(np.mean(monto))

        keys_j = _keys(X["j"])
        self.median_log_j_ = self._group_stat(keys_j, log_monto, "median")
        self.global_median_log_ = float(np.median(log_monto))

        frame = pd.DataFrame({"k": keys_j.to_numpy(), "v": monto})
        self.sorted_by_j_ = {
            key: np.sort(group.to_numpy())
            for key, group in frame.groupby("k")["v"]
            if len(group) >= self.min_group_size
        }
        self.global_sorted_ = np.sort(monto)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calcula las cinco features usando los estadisticos aprendidos."""
        check_is_fitted(self, "global_mean_")
        monto = X["monto"].to_numpy(dtype=float)
        log_monto = X["log_monto"].to_numpy(dtype=float)
        out = pd.DataFrame(index=X.index)

        for name, col in self._RATIO_SPECS:
            mean = (
                _keys(X[col]).map(self.mean_maps_[name]).fillna(self.global_mean_)
            ).to_numpy(dtype=float)
            out[name] = monto / np.where(mean > 0, mean, np.nan)

        keys_j = _keys(X["j"])
        median = (
            keys_j.map(self.median_log_j_).fillna(self.global_median_log_)
        ).to_numpy(dtype=float)
        out["monto_rel_j"] = log_monto - median

        pct = np.empty(len(X), dtype=float)
        positions = pd.Series(np.arange(len(X))).groupby(keys_j.to_numpy()).indices
        for key, pos in positions.items():
            ref = self.sorted_by_j_.get(key, self.global_sorted_)
            pct[pos] = np.searchsorted(ref, monto[pos], side="right") / len(ref)
        out["monto_pct_j"] = pct
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nombres de las columnas de salida."""
        return np.array(
            [name for name, _ in self._RATIO_SPECS] + ["monto_rel_j", "monto_pct_j"]
        )


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Reemplaza cada categoria por su frecuencia relativa en train.

    Genera una columna `frec_<col>` por cada columna de entrada. Las categorias
    no vistas en `fit` reciben 0. Se usa la proporcion y no el conteo para que
    el valor sea comparable entre folds de distinto tamano.

    Args:
        columns: Columnas categoricas a codificar.
    """

    def __init__(self, columns: tuple[str, ...] = ("j", "g")):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        """Aprende la frecuencia relativa de cada categoria (solo con X)."""
        self.freq_maps_ = {
            col: _keys(X[col]).value_counts(normalize=True) for col in self.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Devuelve las frecuencias aprendidas para cada fila."""
        check_is_fitted(self, "freq_maps_")
        out = pd.DataFrame(index=X.index)
        for col in self.columns:
            out[f"frec_{col}"] = (
                _keys(X[col]).map(self.freq_maps_[col]).fillna(0.0).to_numpy(dtype=float)
            )
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nombres de las columnas de salida."""
        return np.array([f"frec_{col}" for col in self.columns])


class RareCategoryGrouper(TransformerMixin, BaseEstimator):
    """Agrupa en `OTRO` las categorias poco frecuentes de una columna.

    Genera la columna `<column>_agrupado`. Los nulos se tratan como la
    categoria `FALTANTE`, que se mantiene solo si supera la frecuencia minima.
    Las categorias no vistas en `fit` pasan a `OTRO`.

    Args:
        column: Columna categorica a agrupar.
        min_frequency: Frecuencia relativa minima para conservar una categoria.
    """

    def __init__(self, column: str = "g", min_frequency: float = 0.001):
        self.column = column
        self.min_frequency = min_frequency

    def fit(self, X: pd.DataFrame, y=None):
        """Aprende que categorias superan la frecuencia minima (solo con X)."""
        freq = _keys(X[self.column]).value_counts(normalize=True)
        self.frequent_ = set(freq[freq >= self.min_frequency].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reemplaza las categorias raras o desconocidas por `OTRO`."""
        check_is_fitted(self, "frequent_")
        keys = _keys(X[self.column])
        grouped = keys.where(keys.isin(self.frequent_), OTHER_LABEL)
        return pd.DataFrame({f"{self.column}_agrupado": grouped}, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nombre de la columna de salida."""
        return np.array([f"{self.column}_agrupado"])


class GroupBehaviorFeatures(TransformerMixin, BaseEstimator):
    """Perfil de comportamiento de cada categoria `j`.

    Columnas de entrada: `j`, `log_monto` y, si `use_score`, `score`.
    Columnas de salida: `j_std_log_monto` (dispersion del monto en la
    categoria) y, con `use_score`, `j_mean_score` y `score_rel_j` (score de la
    fila menos el score medio de su categoria).

    En `fit_transform` los estadisticos de cada fila se calculan sin ella
    (cross-fitting en `n_splits` folds), para que train no vea estadisticos que
    incluyen su propia fila y `transform` (datos nuevos) sea consistente.
    Las categorias con menos de `min_group_size` filas o no vistas usan el
    estadistico global.

    Args:
        min_group_size: Tamano minimo de grupo para usar su propio estadistico.
        n_splits: Folds del cross-fitting en `fit_transform`.
        random_state: Semilla del cross-fitting.
        use_score: Si es False no calcula las features derivadas de `score`.
    """

    def __init__(
        self,
        min_group_size: int = 5,
        n_splits: int = 5,
        random_state: int = config.RANDOM_STATE,
        use_score: bool = True,
    ):
        self.min_group_size = min_group_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.use_score = use_score

    def _learn(self, X: pd.DataFrame):
        """Estadisticos por grupo y globales a partir de X."""
        frame = pd.DataFrame(
            {"k": _keys(X["j"]).to_numpy(), "log_monto": X["log_monto"].to_numpy(float)}
        )
        if self.use_score:
            frame["score"] = X["score"].to_numpy(float)
        grouped = frame.groupby("k")
        stats = pd.DataFrame({"std_log_monto": grouped["log_monto"].std()})
        glob = {"std_log_monto": float(frame["log_monto"].std())}
        if self.use_score:
            stats["mean_score"] = grouped["score"].mean()
            glob["mean_score"] = float(frame["score"].mean())
        return stats[grouped.size() >= self.min_group_size], glob

    def _apply(self, X: pd.DataFrame, stats: pd.DataFrame, glob: dict) -> pd.DataFrame:
        """Mapea los estadisticos aprendidos a las filas de X."""
        keys = _keys(X["j"])
        out = pd.DataFrame(index=X.index)
        out["j_std_log_monto"] = (
            keys.map(stats["std_log_monto"]).fillna(glob["std_log_monto"]).to_numpy(float)
        )
        if self.use_score:
            mean = keys.map(stats["mean_score"]).fillna(glob["mean_score"]).to_numpy(float)
            out["j_mean_score"] = mean
            out["score_rel_j"] = X["score"].to_numpy(float) - mean
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nombres de las columnas de salida."""
        names = ["j_std_log_monto"]
        if self.use_score:
            names += ["j_mean_score", "score_rel_j"]
        return np.array(names)

    def fit(self, X: pd.DataFrame, y=None):
        """Aprende los estadisticos por grupo con todo X."""
        self.stats_, self.glob_ = self._learn(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica los estadisticos aprendidos (datos no vistos en fit)."""
        check_is_fitted(self, "stats_")
        return self._apply(X, self.stats_, self.glob_)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Ajusta con todo X y devuelve las features calculadas por cross-fitting."""
        self.fit(X)
        out = pd.DataFrame(
            np.nan, index=X.index, columns=self.get_feature_names_out(), dtype=float
        )
        kfold = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for fit_idx, held_idx in kfold.split(np.arange(len(X))):
            stats, glob = self._learn(X.iloc[fit_idx])
            out.iloc[held_idx] = self._apply(X.iloc[held_idx], stats, glob).to_numpy()
        return out
