"""Tarea 5: deteccion de anomalias no supervisada.

Los detectores se ajustan sin `fraude` ni `score`. La etiqueta se usa solo
despues de puntuar (`evaluate_detector`), para la comparacion ex-post.

Todos los detectores devuelven un puntaje donde mayor = mas anomalo.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

from src import config
from src.data import load_clean, temporal_split
from src.features import add_basic_features, drop_unused_columns
from src.pipelines import build_anomaly_preprocessor, to_pandas

LIFT_FRACTIONS = (0.01, 0.02, 0.05)


# ---------------------------------------------------------------------------
# Detectores
# ---------------------------------------------------------------------------

class IsolationForestDetector:
    """Isolation Forest: las anomalias se aislan con pocos cortes aleatorios."""

    def __init__(self, n_estimators: int = 300):
        self.model = IsolationForest(
            n_estimators=n_estimators, n_jobs=-1, random_state=config.RANDOM_STATE
        )

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Ajusta el bosque con X."""
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Puntaje de anomalia (mayor = mas anomalo)."""
        return -self.model.score_samples(X)


class _PCABase:
    """PCA que retiene `variance` de la varianza; base de Q y T2."""

    def __init__(self, variance: float = 0.90):
        self.pca = PCA(
            n_components=variance, svd_solver="full", random_state=config.RANDOM_STATE
        )

    def fit(self, X: np.ndarray):
        """Ajusta el PCA con X."""
        self.pca.fit(X)
        return self


class PCAQDetector(_PCABase):
    """Estadistico Q (SPE): error de reconstruccion fuera del subespacio principal."""

    def score(self, X: np.ndarray) -> np.ndarray:
        """Suma de cuadrados del residuo de reconstruccion."""
        recon = self.pca.inverse_transform(self.pca.transform(X))
        return ((X - recon) ** 2).sum(axis=1)


class PCAT2Detector(_PCABase):
    """Estadistico T2 de Hotelling: distancia dentro del subespacio principal."""

    def score(self, X: np.ndarray) -> np.ndarray:
        """Suma de scores al cuadrado normalizados por su varianza."""
        z = self.pca.transform(X)
        return (z**2 / self.pca.explained_variance_).sum(axis=1)


class LOFDetector:
    """Local Outlier Factor en modo novedad, ajustado con una submuestra.

    Args:
        n_neighbors: Vecinos usados para estimar la densidad local.
        max_fit_rows: Maximo de filas de train usadas para ajustar (coste).
    """

    def __init__(self, n_neighbors: int = 35, max_fit_rows: int = 20_000):
        self.max_fit_rows = max_fit_rows
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors, novelty=True, n_jobs=-1
        )

    def fit(self, X: np.ndarray) -> "LOFDetector":
        """Ajusta con una submuestra aleatoria reproducible."""
        if len(X) > self.max_fit_rows:
            rng = np.random.default_rng(config.RANDOM_STATE)
            X = X[rng.choice(len(X), self.max_fit_rows, replace=False)]
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Puntaje de anomalia (mayor = mas anomalo)."""
        return -self.model.score_samples(X)


class KMeansDetector:
    """Distancia al centroide mas cercano de K-Means."""

    def __init__(self, n_clusters: int = 8):
        self.model = KMeans(
            n_clusters=n_clusters, n_init=5, random_state=config.RANDOM_STATE
        )

    def fit(self, X: np.ndarray) -> "KMeansDetector":
        """Ajusta los centroides con X."""
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Distancia al centroide mas cercano."""
        return self.model.transform(X).min(axis=1)


def default_detectors() -> dict:
    """Detectores usados por defecto, indexados por nombre."""
    return {
        "isolation_forest": IsolationForestDetector(),
        "pca_q": PCAQDetector(),
        "pca_t2": PCAT2Detector(),
        "lof": LOFDetector(),
        "kmeans": KMeansDetector(),
    }


# ---------------------------------------------------------------------------
# Ejecucion y evaluacion
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """Salida de `run_anomaly_detection`.

    Attributes:
        scores: Puntajes de test por detector y `ensemble`, mas `id`, `fecha`.
        labels: Etiqueta `fraude` de test (solo para evaluacion ex-post).
        thresholds: Umbral por detector, fijado en calibracion.
        sizes: Filas de train, calibracion y test.
        X_train: Features (sin `fraude`) de train, referencia para explicar.
        X_test: Features (sin `fraude`) de test, alineadas con `scores`.
    """

    scores: pd.DataFrame
    labels: np.ndarray
    thresholds: dict[str, float]
    sizes: tuple[int, int, int]
    X_train: pd.DataFrame
    X_test: pd.DataFrame


def _prepare(df: pl.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Agrega features y separa X de y. `fraude` sale de X y se guarda aparte."""
    pdf = to_pandas(drop_unused_columns(add_basic_features(df)))
    y = pdf[config.TARGET_COLUMN].to_numpy()
    return pdf.drop(columns=[config.TARGET_COLUMN]), y


def _ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Posicion relativa (0 a 1) de `values` dentro de la distribucion `reference`."""
    return np.searchsorted(np.sort(reference), values, side="right") / len(reference)


def run_anomaly_detection(
    alert_rate: float = config.ANOMALY_ALERT_RATE, detectors: dict | None = None
) -> AnomalyResult:
    """Ajusta los detectores en train, fija umbrales en calibracion y puntua test.

    Split temporal 60/20/20. El umbral de cada detector es el cuantil
    `1 - alert_rate` de sus puntajes en calibracion. El `ensemble` promedia la
    posicion de cada puntaje dentro de la distribucion de calibracion.
    `fraude` no entra al ajuste ni a la eleccion de umbrales.

    Args:
        alert_rate: Proporcion de transacciones a marcar como sospechosas.
        detectors: Detectores a usar. Por defecto `default_detectors()`.

    Returns:
        `AnomalyResult` con puntajes, etiquetas y umbrales.
    """
    detectors = detectors or default_detectors()
    train, cal, test = temporal_split(load_clean())
    (X_train, _), (X_cal, _), (X_test, y_test) = (_prepare(d) for d in (train, cal, test))

    prep = build_anomaly_preprocessor().fit(X_train)
    M_train, M_cal, M_test = (prep.transform(X) for X in (X_train, X_cal, X_test))

    cal_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    for name, det in detectors.items():
        det.fit(M_train)
        cal_scores[name] = det.score(M_cal)
        test_scores[name] = det.score(M_test)

    thresholds = {
        name: float(np.quantile(s, 1 - alert_rate)) for name, s in cal_scores.items()
    }
    test_scores["ensemble"] = np.mean(
        [_ecdf(cal_scores[n], test_scores[n]) for n in detectors], axis=0
    )
    cal_ensemble = np.mean(
        [_ecdf(cal_scores[n], cal_scores[n]) for n in detectors], axis=0
    )
    thresholds["ensemble"] = float(np.quantile(cal_ensemble, 1 - alert_rate))

    scores = pd.DataFrame(test_scores)
    scores.insert(0, config.DATE_COLUMN, X_test[config.DATE_COLUMN].to_numpy())
    scores.insert(0, config.ID_COLUMN, X_test[config.ID_COLUMN].to_numpy())
    return AnomalyResult(
        scores,
        y_test,
        thresholds,
        (train.height, cal.height, test.height),
        X_train,
        X_test,
    )


def evaluate_detector(
    y_true: np.ndarray, score: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compara un detector con `fraude` (analisis ex-post).

    Args:
        y_true: Etiqueta real de fraude.
        score: Puntaje de anomalia (mayor = mas anomalo).
        threshold: Umbral de alerta fijado en calibracion.

    Returns:
        Diccionario con tasa base, PR-AUC, ROC-AUC, lift en el top k% y
        precision/recall/alertas con el umbral dado.
    """
    y_true = np.asarray(y_true)
    base = float(y_true.mean())
    order = np.argsort(-score)
    out = {
        "base_rate": base,
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
    }
    for frac in LIFT_FRACTIONS:
        k = max(1, int(len(score) * frac))
        out[f"lift@{int(frac * 100)}%"] = float(y_true[order[:k]].mean() / base)
    flagged = score >= threshold
    out["alerts"] = int(flagged.sum())
    out["precision"] = float(y_true[flagged].mean()) if flagged.any() else 0.0
    out["recall"] = float(y_true[flagged].sum() / y_true.sum())
    return out


# ---------------------------------------------------------------------------
# Explicacion de casos
# ---------------------------------------------------------------------------

# `j` se excluye: con ~7.600 categorias toda categoria es rara y no distingue.
EXPLAIN_CATEGORICAL = ("o", "hora_bin", "g")
# Codificaciones ciclicas: duplican a `hora` y no son legibles como razon.
EXPLAIN_EXCLUDED_NUMERIC = ("hora_sin", "hora_cos", "dow_sin", "dow_cos")
CONTEXT_COLUMNS = ("id", "fecha", "monto", "g", "j", "i", "o", "hora")


def _explain_numeric_columns() -> list[str]:
    """Numericas interpretables, sin `score` ni derivadas."""
    excluded = {"score", *config.SCORE_DERIVED_FEATURES, *EXPLAIN_EXCLUDED_NUMERIC}
    return [c for c in config.NUMERIC_FEATURES if c not in excluded]


def rarity_matrix(X_ref: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de observar un valor tan extremo, segun la referencia.

    Numericas: cola mas chica entre P(V <= v) y P(V >= v) en `X_ref` (maneja
    empates en variables binarias o discretas). Categoricas: frecuencia de la
    categoria en `X_ref` (0 si no se vio). Los nulos numericos quedan en NaN.
    Menor valor = mas raro.

    Args:
        X_ref: Features de referencia (train).
        X: Features a explicar.

    Returns:
        DataFrame con una columna por feature y una fila por fila de `X`.
    """
    out = {}
    for col in _explain_numeric_columns():
        ref = np.sort(X_ref[col].dropna().to_numpy(dtype=float))
        v = X[col].to_numpy(dtype=float)
        low = np.searchsorted(ref, v, side="right") / len(ref)
        high = 1 - np.searchsorted(ref, v, side="left") / len(ref)
        out[col] = np.where(np.isnan(v), np.nan, np.minimum(low, high))
    for col in EXPLAIN_CATEGORICAL:
        freq = X_ref[col].fillna("FALTANTE").astype(str).value_counts(normalize=True)
        out[col] = (
            X[col].fillna("FALTANTE").astype(str).map(freq).fillna(0.0).to_numpy()
        )
    return pd.DataFrame(out, index=X.index)


def _top_reasons(rarity: pd.DataFrame, n_reasons: int) -> np.ndarray:
    """Indices de las `n_reasons` features mas raras de cada fila."""
    values = rarity.to_numpy(dtype=float)
    return np.argsort(np.where(np.isnan(values), np.inf, values), axis=1)[:, :n_reasons]


def explain_cases(
    result: AnomalyResult,
    score_col: str = "isolation_forest",
    top_n: int = 200,
    n_reasons: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explica los `top_n` casos mas anomalos segun `score_col`.

    Para cada caso lista las `n_reasons` variables con el valor mas raro frente
    a train. Ademas resume cuanto aparece cada variable entre los casos
    marcados frente a todo el test, para distinguir lo realmente distintivo de
    lo que es raro en cualquier fila. `fraude` solo se adjunta como columna
    ex-post; no interviene en las razones.

    Args:
        result: Salida de `run_anomaly_detection`.
        score_col: Detector a explicar (columna de `result.scores`).
        top_n: Cantidad de casos a explicar.
        n_reasons: Variables explicativas por caso.

    Returns:
        Tupla (casos, resumen). `casos` tiene contexto, puntaje, razones y
        `fraude`. `resumen` tiene, por variable, la frecuencia como razon en
        los casos marcados y en todo el test, y su cociente.
    """
    rarity = rarity_matrix(result.X_train, result.X_test)
    names = np.array(rarity.columns)
    reasons = _top_reasons(rarity, n_reasons)

    scores = result.scores[score_col].to_numpy()
    top = np.argsort(-scores)[:top_n]

    def describe(row: int) -> str:
        parts = []
        for j in reasons[row]:
            col = names[j]
            value = result.X_test.iloc[row][col]
            shown = f"{value:.3g}" if isinstance(value, (int, float, np.number)) else value
            parts.append(f"{col}={shown} (p={rarity.iloc[row, j]:.2%})")
        return "; ".join(parts)

    context = [c for c in CONTEXT_COLUMNS if c in result.X_test.columns]
    cases = result.X_test.iloc[top][context].copy()
    cases["puntaje"] = scores[top]
    cases["razones"] = [describe(r) for r in top]
    cases[config.TARGET_COLUMN] = result.labels[top]

    def share(idx: np.ndarray) -> pd.Series:
        counts = np.bincount(reasons[idx].ravel(), minlength=len(names))
        return pd.Series(counts / len(idx), index=names)

    summary = pd.DataFrame(
        {
            "en_casos_marcados": share(top),
            "en_todo_test": share(np.arange(len(scores))),
        }
    )
    summary["cociente"] = summary["en_casos_marcados"] / summary["en_todo_test"].clip(lower=1e-9)
    return cases.reset_index(drop=True), summary.sort_values("en_casos_marcados", ascending=False)
