"""Modulo de explicabilidad e interpretabilidad para el modelo ganador (LightGBM).

Genera importancias globales (Gain, Split, SHAP) y explicaciones locales
de transacciones individuales usando TreeSHAP nativo de LightGBM (sin dependencias extra).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config


def clean_feature_name(name: str) -> str:
    """Limpia los prefijos generados por ColumnTransformer ('num__', 'group__', etc.)."""
    if "__" in name:
        return name.split("__", 1)[1]
    return name


def get_feature_names(preprocessor) -> list[str]:
    """Obtiene los nombres legibles de las variables transformadas."""
    try:
        raw_names = preprocessor.get_feature_names_out()
        return [clean_feature_name(n) for n in raw_names]
    except Exception:
        # Fallback si algun transformador no implementa get_feature_names_out
        return [f"feature_{i}" for i in range(preprocessor.n_features_in_)]


def compute_shap_and_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_size: int = 15_000,
    seed: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Calcula importancias globales y contribuciones SHAP.

    Args:
        pipeline: Pipeline ajustado (prep + model).
        X: Variables predictoras (train o evaluacion).
        y: Etiquetas reales.
        sample_size: Cantidad de filas para estimar SHAP (LightGBM nativo).
        seed: Semilla para el muestreo representativo.

    Returns:
        Tupla (df_importancia, shap_values_sample, feature_names).
    """
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    # Transformacion de X
    X_trans = prep.transform(X)
    feature_names = get_feature_names(prep)

    # Importancia nativa de LightGBM (gain y split)
    booster = model.booster_
    gain_imp = booster.feature_importance(importance_type="gain")
    split_imp = booster.feature_importance(importance_type="split")

    # Muestreo estratificado para SHAP
    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        n_pos = int(sample_size * (len(idx_pos) / len(y)))
        n_neg = sample_size - n_pos
        sampled_pos = rng.choice(idx_pos, size=n_pos, replace=False)
        sampled_neg = rng.choice(idx_neg, size=n_neg, replace=False)
        sample_idx = np.concatenate([sampled_pos, sampled_neg])
        rng.shuffle(sample_idx)
        X_sample = X_trans[sample_idx]
    else:
        X_sample = X_trans

    # LightGBM TreeSHAP nativo: devuelve matriz (n_samples, n_features + 1)
    # La ultima columna es el valor base (expected value en espacio margen/logit).
    shap_matrix = booster.predict(X_sample, pred_contrib=True)
    shap_features = shap_matrix[:, :-1]
    base_value = float(shap_matrix[0, -1])

    # Estadisticas globales de SHAP
    mean_abs_shap = np.mean(np.abs(shap_features), axis=0)
    mean_shap = np.mean(shap_features, axis=0)

    df_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
            "gain_importance": gain_imp,
            "split_importance": split_imp,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Porcentaje de contribucion
    df_importance["pct_importance"] = (
        df_importance["mean_abs_shap"] / df_importance["mean_abs_shap"].sum()
    ) * 100

    return df_importance, shap_features, feature_names


def explain_instance(
    pipeline: Pipeline,
    row_df: pd.DataFrame,
    top_k: int = 5,
) -> dict:
    """Genera la explicacion local de una transaccion individual.

    Args:
        pipeline: Pipeline ajustado.
        row_df: DataFrame con 1 fila conteniendo las columnas de entrada.
        top_k: Cantidad de factores principales a reportar.

    Returns:
        Diccionario con proba, contribuciones positivas y negativas.
    """
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(prep)

    X_trans = prep.transform(row_df)
    proba = float(pipeline.predict_proba(row_df)[0, 1])

    shap_matrix = model.booster_.predict(X_trans, pred_contrib=True)
    shap_row = shap_matrix[0, :-1]
    base_val = float(shap_matrix[0, -1])

    order = np.argsort(np.abs(shap_row))[::-1]
    top_features = []
    for idx in order[:top_k]:
        feat = feature_names[idx]
        val_contrib = float(shap_row[idx])
        top_features.append(
            {
                "feature": feat,
                "shap_contribution": val_contrib,
                "direction": "Incrementa Riesgo" if val_contrib > 0 else "Disminuye Riesgo",
            }
        )

    return {
        "predicted_probability": proba,
        "base_value_logit": base_val,
        "top_drivers": top_features,
    }
