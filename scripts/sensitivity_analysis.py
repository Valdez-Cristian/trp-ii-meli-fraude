"""Analisis de sensibilidad economica y cuantificacion de ahorro de negocio (Tarea 4)."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

from src import config
from src.data import load_train
from src.optimization import threshold_table, best_threshold

def run_sensitivity():
    train = load_train()
    y_true = train[config.TARGET_COLUMN].to_numpy()
    total_fraud = int(y_true.sum())
    total_legit = int(len(y_true) - total_fraud)

    out_dir = config.PROCESSED_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar probabilidades OOF del modelo ganador y score legado
    oof_lgb = pl.read_parquet(config.OOF_DIR / "lightgbm_tuned.parquet")["proba"].to_numpy()
    legacy_score = train["score"].to_numpy() / 100.0

    print("=" * 80)
    print("TAREA 4: ANÁLISIS DE SENSIBILIDAD FINANCIERA Y CASO DE NEGOCIO")
    print("=" * 80)

    # 1. Barrido de sensibilidad de Costo de Falsos Positivos
    # Evaluamos costos de FP desde $1 hasta $50 (friccion moderada a alta/churn)
    fp_cost_grid = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    sweep_results = []

    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)

    for cost_fp in fp_cost_grid:
        # a) Sin restriccion
        tbl = threshold_table(y_true, oof_lgb, thresholds=thresholds, cost_fp=cost_fp, cost_fn=config.COSTO_FN)
        best_uncon_idx = tbl["cost"].idxmin()
        best_uncon = tbl.loc[best_uncon_idx]

        # b) Con restriccion de capacidad FP_MAX (2,137)
        tbl_con = tbl[tbl["fp"] <= config.FP_MAX]
        best_con_idx = tbl_con["cost"].idxmin()
        best_con = tbl_con.loc[best_con_idx]

        sweep_results.append({
            "cost_fp": cost_fp,
            "cost_fn": config.COSTO_FN,
            "ratio_fn_fp": config.COSTO_FN / cost_fp,
            # Sin restriccion
            "thresh_uncon": best_uncon.threshold,
            "cost_uncon": best_uncon.cost,
            "fp_uncon": int(best_uncon.fp),
            "fn_uncon": int(best_uncon.fn),
            "recall_uncon": (total_fraud - best_uncon.fn) / total_fraud,
            # Con restriccion
            "thresh_con": best_con.threshold,
            "cost_con": best_con.cost,
            "fp_con": int(best_con.fp),
            "fn_con": int(best_con.fn),
            "recall_con": (total_fraud - best_con.fn) / total_fraud,
        })

    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(out_dir / "cost_sensitivity_sweep.csv", index=False)

    print("\n1. IMPACTO DE VARIACIÓN EN COSTO DE FP ($1 a $50):")
    for r in sweep_results:
        print(
            f"  * Costo FP: ${r['cost_fp']:>2.0f} | Ratio FN:FP: {r['ratio_fn_fp']:>3.0f}:1 | "
            f"Sin rest: Umbral {r['thresh_uncon']:.2f}, FP {r['fp_uncon']:>5}, Costo ${r['cost_uncon']:>9,.0f} | "
            f"Con rest: Umbral {r['thresh_con']:.2f}, FP {r['fp_con']:>5}, Costo ${r['cost_con']:>9,.0f}"
        )

    # 2. Comparacion formal del Caso de Negocio (ROI y Ahorro)
    # Escenario A: No hacer nada
    cost_no_action = total_fraud * config.COSTO_FN
    
    # Escenario B: Score legado bajo restriccion FP_MAX
    tbl_leg = threshold_table(y_true, legacy_score, thresholds=thresholds, cost_fp=config.COSTO_FP, cost_fn=config.COSTO_FN)
    tbl_leg_con = tbl_leg[tbl_leg["fp"] <= config.FP_MAX]
    leg_con = tbl_leg_con.loc[tbl_leg_con["cost"].idxmin()]
    cost_legacy = leg_con.cost
    rec_legacy = (total_fraud - leg_con.fn) / total_fraud
    prec_legacy = (total_fraud - leg_con.fn) / (total_fraud - leg_con.fn + leg_con.fp)

    # Escenario C: LightGBM Tuned bajo restriccion FP_MAX
    lgb_row = df_sweep[df_sweep["cost_fp"] == config.COSTO_FP].iloc[0]
    cost_lgb = lgb_row["cost_con"]
    rec_lgb = lgb_row["recall_con"]
    prec_lgb = (total_fraud - lgb_row["fn_con"]) / (total_fraud - lgb_row["fn_con"] + lgb_row["fp_con"])

    savings_vs_none = cost_no_action - cost_lgb
    pct_vs_none = (savings_vs_none / cost_no_action) * 100

    savings_vs_legacy = cost_legacy - cost_lgb
    pct_vs_legacy = (savings_vs_legacy / cost_legacy) * 100

    business_cases = [
        {
            "Escenario": "1. No hacer nada (sin filtros)",
            "Umbral": "N/A",
            "Costo Total ($)": cost_no_action,
            "FP": 0,
            "FN": total_fraud,
            "Precision": 0.0,
            "Recall": 0.0,
            "Ahorro vs No Acción ($)": 0.0,
            "Ahorro vs No Acción (%)": 0.0,
            "Ahorro vs Legado ($)": -(cost_no_action - cost_legacy),
        },
        {
            "Escenario": "2. Motor actual (Score Legado >= 98)",
            "Umbral": f"Score >= {int(leg_con.threshold * 100)}",
            "Costo Total ($)": cost_legacy,
            "FP": int(leg_con.fp),
            "FN": int(leg_con.fn),
            "Precision": prec_legacy,
            "Recall": rec_legacy,
            "Ahorro vs No Acción ($)": cost_no_action - cost_legacy,
            "Ahorro vs No Acción (%)": ((cost_no_action - cost_legacy) / cost_no_action) * 100,
            "Ahorro vs Legado ($)": 0.0,
        },
        {
            "Escenario": "3. Modelo Propuesto (LightGBM Tuned >= 0.26)",
            "Umbral": f"Proba >= {lgb_row['thresh_con']:.2f}",
            "Costo Total ($)": cost_lgb,
            "FP": int(lgb_row["fp_con"]),
            "FN": int(lgb_row["fn_con"]),
            "Precision": prec_lgb,
            "Recall": rec_lgb,
            "Ahorro vs No Acción ($)": savings_vs_none,
            "Ahorro vs No Acción (%)": pct_vs_none,
            "Ahorro vs Legado ($)": savings_vs_legacy,
        },
    ]

    df_cases = pd.DataFrame(business_cases)
    df_cases.to_csv(out_dir / "business_case_comparison.csv", index=False)

    print("\n2. CUADRO DE IMPACTO ECONÓMICO Y AHORRO EMPRESARIAL:")
    print("-" * 80)
    for c in business_cases:
        print(f"[{c['Escenario']}]")
        print(f"  * Costo Total: ${c['Costo Total ($)']:,.0f} | FP: {c['FP']:,} | FN: {c['FN']:,} | Prec: {c['Precision']:.1%} | Rec: {c['Recall']:.1%}")
        if c['Ahorro vs Legado ($)'] > 0:
            print(f"  * AHORRO NETO FRENTE AL SISTEMA ACTUAL: ${c['Ahorro vs Legado ($)']:,.0f} ({pct_vs_legacy:.1f}% de reduccion de perdidas)")
        elif c['Ahorro vs No Acción ($)'] > 0:
            print(f"  * Reducción de costos frente a no hacer nada: ${c['Ahorro vs No Acción ($)']:,.0f}")
    print("=" * 80)

if __name__ == "__main__":
    run_sensitivity()
