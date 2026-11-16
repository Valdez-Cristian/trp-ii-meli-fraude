"""Script de ejecucion de interpretabilidad y explicabilidad SHAP (Tarea 3 y 4)."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

from src import config
from src.data import load_train
from src.explain import compute_shap_and_importance, explain_instance
from src.models import get_model
from src.pipelines import build_pipeline, split_xy

def main():
    out_dir = config.PROCESSED_DIR / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AJUSTE DE MODELO GANADOR (LightGBM Tuned) Y CÁLCULO DE EXPLICABILIDAD SHAP")
    print("=" * 80)

    train_df = load_train()
    X, y = split_xy(train_df)

    kind, estimator = get_model("lightgbm_tuned")
    pipeline = build_pipeline(kind, estimator, use_score=True)

    print("Ajustando pipeline en train...")
    pipeline.fit(X, y)

    print("Calculando TreeSHAP e importancias de features...")
    df_imp, shap_sample, feat_names = compute_shap_and_importance(
        pipeline, X, y, sample_size=15_000, seed=config.RANDOM_STATE
    )

    df_imp.to_csv(out_dir / "global_importance.csv", index=False)

    print("\nTOP 15 FEATURES MÁS DETERMINANTES (por impacto absoluto medio SHAP):")
    for idx, row in df_imp.head(15).iterrows():
        print(
            f"  {idx+1:2d}. {row['feature']:<25} | Importancia SHAP: {row['mean_abs_shap']:.4f} "
            f"({row['pct_importance']:>4.1f}%) | Gain: {row['gain_importance']:>9,.0f} | "
            f"Efecto promedio: {row['mean_shap']:+.4f}"
        )

    # Explicacion de casos representativos
    # Cargar OOF para ver clasificaciones con el umbral 0.26
    thresh_info = json.loads((config.OOF_DIR / "lightgbm_tuned_threshold.json").read_text())
    thresh = thresh_info["threshold"]

    oof_proba = pl.read_parquet(config.OOF_DIR / "lightgbm_tuned.parquet")["proba"].to_numpy()
    preds = oof_proba >= thresh

    # Identificar indices de TP, FP, FN y TN
    tp_indices = np.where((y == 1) & (preds == True))[0]
    fp_indices = np.where((y == 0) & (preds == True))[0]
    fn_indices = np.where((y == 1) & (preds == False))[0]

    examples = {}
    if len(tp_indices) > 0:
        idx_tp = int(tp_indices[0])
        exp_tp = explain_instance(pipeline, X.iloc[[idx_tp]], top_k=5)
        exp_tp["tipo"] = "True Positive (Fraude Detectado Correctamente)"
        exp_tp["id"] = int(X.iloc[idx_tp]["id"]) if "id" in X.columns else idx_tp
        examples["TP"] = exp_tp

    if len(fp_indices) > 0:
        idx_fp = int(fp_indices[0])
        exp_fp = explain_instance(pipeline, X.iloc[[idx_fp]], top_k=5)
        exp_fp["tipo"] = "False Positive (Falsa Alarma / Fricción)"
        exp_fp["id"] = int(X.iloc[idx_fp]["id"]) if "id" in X.columns else idx_fp
        examples["FP"] = exp_fp

    if len(fn_indices) > 0:
        idx_fn = int(fn_indices[0])
        exp_fn = explain_instance(pipeline, X.iloc[[idx_fn]], top_k=5)
        exp_fn["tipo"] = "False Negative (Fraude Escapado / Pérdida $200)"
        exp_fn["id"] = int(X.iloc[idx_fn]["id"]) if "id" in X.columns else idx_fn
        examples["FN"] = exp_fn

    (out_dir / "explained_examples.json").write_text(json.dumps(examples, indent=2))

    print("\nEJEMPLOS LOCALES EXPLICADOS GUARDADOS:")
    for key, ex in examples.items():
        print(f"\n[{key}] {ex['tipo']} - Probabilidad: {ex['predicted_probability']:.3f}")
        for drv in ex["top_drivers"]:
            print(f"    - {drv['feature']}: {drv['shap_contribution']:+.3f} ({drv['direction']})")

    print("\nExplicabilidad completada y persistida en data/processed/explain/.")
    print("=" * 80)

if __name__ == "__main__":
    main()
