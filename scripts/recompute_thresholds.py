"""Recalcula los umbrales optimos OOF respetando la restriccion operativa de FP_MAX."""

import json
from pathlib import Path
import numpy as np
import polars as pl

from src import config
from src.data import load_train
from src.metrics import evaluate
from src.optimization import best_threshold

def main():
    train = load_train()
    y_true = train[config.TARGET_COLUMN].to_numpy()
    total_fraud = int(y_true.sum())
    total_legit = int(len(y_true) - total_fraud)

    print("=" * 80)
    print(f"REOPTIMIZACIÓN DE UMBRALES CON RESTRICCIÓN OPERATIVA")
    print(f"Train: {len(y_true):,} transacciones | Fraudes: {total_fraud:,} ({total_fraud/len(y_true):.2%}) | Legítimas: {total_legit:,}")
    print(f"Restricción de Negocio: FP_MAX = {config.FP_MAX:,} (~{config.FP_MAX/total_legit:.1%} de las legítimas, ~50 revisiones/día)")
    print("=" * 80)

    models = ["logistic", "random_forest", "lightgbm", "lightgbm_tuned"]
    summary = []

    for name in models:
        parquet_path = config.OOF_DIR / f"{name}.parquet"
        if not parquet_path.exists():
            continue
        oof = pl.read_parquet(parquet_path)
        proba = oof["proba"].to_numpy()

        # Umbral con FP_MAX
        thresh = best_threshold(y_true, proba, fp_max=config.FP_MAX)
        metrics = evaluate(y_true, proba, threshold=thresh)

        # Guardar threshold json actualizado
        thresh_path = config.OOF_DIR / f"{name}_threshold.json"
        thresh_path.write_text(
            json.dumps({"threshold": thresh, "fp_max": config.FP_MAX}, indent=2)
        )

        row = {
            "Modelo": name,
            "Umbral": thresh,
            "PR-AUC": metrics["pr_auc"],
            "Costo": metrics["cost"],
            "FP": int(metrics["fp"]),
            "FN": int(metrics["fn"]),
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Alertas": int(metrics["fp"] + total_fraud - metrics["fn"]),
        }
        summary.append(row)

    df_summary = pl.DataFrame(summary).sort("Costo")
    print("\nRESUMEN COMPARATIVO (OOF sobre train, 5 folds):")
    for r in df_summary.iter_rows(named=True):
        print(
            f"  * {r['Modelo']:<16} | Umbral: {r['Umbral']:.2f} | Costo: ${r['Costo']:>9,.0f} | "
            f"PR-AUC: {r['PR-AUC']:.4f} | FP: {r['FP']:>5} | FN: {r['FN']:>5} | "
            f"Prec: {r['Precision']:>5.1%} | Rec: {r['Recall']:>5.1%} | Alertas: {r['Alertas']:>5}"
        )
    print("=" * 80)

if __name__ == "__main__":
    main()
