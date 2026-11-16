"""Evaluacion ciega final sobre el conjunto de test (25%).

Este script es el unico punto autorizado del proyecto para llamar a `load_test()`.

"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config
from src.data import load_test, load_train
from src.metrics import confusion_counts, evaluate
from src.models import get_model
from src.pipelines import build_pipeline, split_xy


def main():
    print("=" * 80)
    print("EVALUACIÓN CIEGA FINAL EN CONJUNTO DE TEST (25%)")
    print("=" * 80)

    # 1. Cargar train y ajustar el pipeline ganador
    train_df = load_train()
    X_train, y_train = split_xy(train_df)

    # lightgbm_tuned | lightgbm | random_forest | logistic
    model: str = "lightgbm_tuned"

    kind, estimator = get_model(model)
    pipeline = build_pipeline(kind, estimator, use_score=True)

    print(f"1. Ajustando modelo final ({model}) sobre {len(X_train):,} transacciones de train...")
    pipeline.fit(X_train, y_train)

    # 2. Cargar umbral optimo congelado
    thresh_file = config.OOF_DIR / f"{model}_threshold.json"
    thresh_data = json.loads(thresh_file.read_text())
    threshold = float(thresh_data["threshold"])
    print(f"2. Umbral congelado de validación cruzada: {threshold:.2f} (con restricción FP_MAX={thresh_data.get('fp_max')})")

    # 3. Cargar test (evaluacion ciega)
    test_df = load_test()
    X_test, y_test = split_xy(test_df)
    total_test = len(y_test)
    total_fraud_test = int(y_test.sum())
    total_legit_test = int(total_test - total_fraud_test)

    print(f"\n3. Datos de Test cargados: {total_test:,} transacciones")
    print(f"   * Fraudes reales: {total_fraud_test:,} ({total_fraud_test / total_test:.2%})")
    print(f"   * Transacciones legítimas: {total_legit_test:,} ({total_legit_test / total_test:.2%})")

    # 4. Inferencia sobre test
    print("\n4. Generando predicciones sobre test...")
    proba_test = pipeline.predict_proba(X_test)[:, 1]

    # Guardar predicciones
    pl.DataFrame(
        {
            config.ID_COLUMN: X_test[config.ID_COLUMN].to_numpy(),
            config.TARGET_COLUMN: y_test,
            "proba": proba_test,
        }
    ).write_parquet(config.PROCESSED_DIR / "test_predictions.parquet")

    # 5. Métricas del Modelo Propuesto
    pred_test = proba_test >= threshold
    counts = confusion_counts(y_test, pred_test)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    pr_auc = float(average_precision_score(y_test, proba_test))
    roc_auc = float(roc_auc_score(y_test, proba_test))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    cost_total = float(fp * config.COSTO_FP + fn * config.COSTO_FN)
    alert_count = int(tp + fp)
    alert_rate = float(alert_count / total_test)
    fpr = float(fp / total_legit_test)

    # 6. Benchmark en Test: No hacer nada y Score Legado
    cost_no_action = float(total_fraud_test * config.COSTO_FN)

    # Score legado con corte train (score >= 98)
    score_test = test_df["score"].to_numpy() / 100.0
    legacy_pred = score_test >= 0.98
    tp_leg = int((legacy_pred & (y_test == 1)).sum())
    fp_leg = int((legacy_pred & (y_test == 0)).sum())
    fn_leg = int((~legacy_pred & (y_test == 1)).sum())
    cost_legacy = float(fp_leg * config.COSTO_FP + fn_leg * config.COSTO_FN)
    rec_legacy = float(tp_leg / total_fraud_test)
    prec_legacy = float(tp_leg / (tp_leg + fp_leg)) if (tp_leg + fp_leg) > 0 else 0.0

    savings_vs_none = cost_no_action - cost_total
    pct_vs_none = (savings_vs_none / cost_no_action) * 100.0

    savings_vs_legacy = cost_legacy - cost_total
    pct_vs_legacy = (savings_vs_legacy / cost_legacy) * 100.0

    # Consolidar resultados
    eval_results = {
        "dataset_split": "test",
        "sample_count": total_test,
        "frauds_count": total_fraud_test,
        "legit_count": total_legit_test,
        "model": model,
        "threshold": threshold,
        "metrics": {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "alert_count": alert_count,
            "alert_rate": alert_rate,
            "fpr": fpr,
            "cost_total": cost_total,
        },
        "business_comparison": {
            "no_action_cost": cost_no_action,
            "legacy_score_cost": cost_legacy,
            "proposed_model_cost": cost_total,
            "savings_vs_no_action": savings_vs_none,
            "savings_vs_no_action_pct": pct_vs_none,
            "savings_vs_legacy": savings_vs_legacy,
            "savings_vs_legacy_pct": pct_vs_legacy,
            "legacy_metrics": {
                "threshold": "score >= 98",
                "tp": tp_leg,
                "fp": fp_leg,
                "fn": fn_leg,
                "precision": prec_legacy,
                "recall": rec_legacy,
            },
        },
    }

    (config.PROCESSED_DIR / "test_evaluation.json").write_text(
        json.dumps(eval_results, indent=2)
    )

    print("\n" + "=" * 80)
    print("RESULTADOS FINALES EN TEST (MÉTRICAS OFICIALES CIEGAS):")
    print("=" * 80)
    print(f"  * PR-AUC (Métrica principal): {pr_auc:.4f}")
    print(f"  * ROC-AUC:                   {roc_auc:.4f}")
    print(f"  * Precisión:                 {precision:.2%}")
    print(f"  * Recall:                    {recall:.2%} ({tp:,} de {total_fraud_test:,} fraudes detectados)")
    print(f"  * Falsos Positivos (FP):     {fp:,} ({fpr:.2%} de transacciones legítimas alertadas)")
    print(f"  * Falsos Negativos (FN):     {fn:,} fraudes no detectados")
    print(f"  * Alertas Totales a Revisión:{alert_count:,} ({alert_rate:.2%} del tráfico)")
    print(f"  * Costo Total Financiero:    ${cost_total:,.0f}")
    print("-" * 80)
    print("IMPACTO FINANCIERO FRENTE A BENCHMARKS EN TEST:")
    print(f"  * Costo 'No hacer nada':     ${cost_no_action:,.0f}")
    print(f"  * Costo 'Score Legado':      ${cost_legacy:,.0f} (Recall {rec_legacy:.1%}, Prec {prec_legacy:.1%})")
    print(f"  * Costo Modelo Propuesto:    ${cost_total:,.0f} (Recall {recall:.1%}, Prec {precision:.1%})")
    print(f"  >>> AHORRO NETO VS LEGADO:   ${savings_vs_legacy:,.0f} ({pct_vs_legacy:.1f}% de reducción de costos)")
    print(f"  >>> AHORRO VS NO ACCIÓN:     ${savings_vs_none:,.0f} ({pct_vs_none:.1f}% de reducción de costos)")
    print("=" * 80)


if __name__ == "__main__":
    main()
