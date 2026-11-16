"""Tarea 5: detecta anomalias sin `fraude` ni `score` y las compara ex-post."""

import logging

import pandas as pd
import polars as pl

from src import config
from src.anomalies import evaluate_detector, explain_cases, run_anomaly_detection

TOP_REVIEW = 800 # incorporamos todas al .csv
EXPLAINED_DETECTOR = "isolation_forest"


def main() -> None:
    """Corre los detectores, imprime la comparacion y guarda los casos a revisar."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_anomaly_detection()
    train_n, cal_n, test_n = result.sizes
    print(f"Split temporal: train {train_n}, calibracion {cal_n}, test {test_n}")
    print(f"Tasa de alerta: {config.ANOMALY_ALERT_RATE:.0%}")

    rows = {
        name: evaluate_detector(result.labels, result.scores[name].to_numpy(), thr)
        for name, thr in result.thresholds.items()
    }
    table = pd.DataFrame(rows).T
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 200
    print(table.to_string())

    config.ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
    scored = result.scores.assign(**{config.TARGET_COLUMN: result.labels})
    pl.from_pandas(scored).write_parquet(config.ANOMALY_DIR / "test_scores.parquet")
    top = scored.sort_values("ensemble", ascending=False).head(TOP_REVIEW)
    pl.from_pandas(top).write_parquet(config.ANOMALY_DIR / "casos_a_revisar.parquet")
    table.to_csv(config.ANOMALY_DIR / "comparacion_detectores.csv")

    cases, summary = explain_cases(result, EXPLAINED_DETECTOR, TOP_REVIEW)
    pl.from_pandas(cases).write_parquet(config.ANOMALY_DIR / "casos_explicados.parquet")
    cases.to_csv(config.ANOMALY_DIR / "casos_explicados.csv", index=False)
    summary.to_csv(config.ANOMALY_DIR / "razones_resumen.csv")

    n_fraud = int(cases[config.TARGET_COLUMN].sum())
    print(f"\nTop {TOP_REVIEW} de {EXPLAINED_DETECTOR}: {n_fraud} fraudes etiquetados, "
          f"{TOP_REVIEW - n_fraud} sin etiqueta (candidatos a fraude nuevo)")
    print("\nVariables que mas explican las alertas:")
    print(summary.head(12).to_string())
    pd.options.display.max_colwidth = 140
    print("\nEjemplos (sin etiqueta de fraude):")
    sin_etiqueta = cases[cases[config.TARGET_COLUMN] == 0]
    print(sin_etiqueta[["id", "monto", "g", "puntaje", "razones"]].head(5).to_string())


if __name__ == "__main__":
    main()
