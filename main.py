"""Orquestador principal del proyecto de Detección de Fraude."""

import argparse
from scripts.prepare_data import main as prepare_data
from scripts.recompute_thresholds import main as recompute_thresholds
from scripts.explain_model import main as explain_model
from scripts.sensitivity_analysis import run_sensitivity as sensitivity_analysis
from scripts.detect_anomalies import main as detect_anomalies
from scripts.final_eval import main as final_eval


def run_all() -> None:
    """Ejecuta todo el flujo del proyecto en secuencia."""
    print("\n>>> PASO 1: Ingesta y partición de datos...")
    prepare_data()

    print("\n>>> PASO 2: Optimización de umbrales bajo FP_MAX...")
    recompute_thresholds()

    print("\n>>> PASO 3: Detección no supervisada de anomalías (Tarea 5)...")
    detect_anomalies()

    print("\n>>> PASO 4: Interpretabilidad TreeSHAP...")
    explain_model()

    print("\n>>> PASO 5: Análisis de sensibilidad financiera...")
    sensitivity_analysis()

    print("\n>>> PASO 6: Evaluación ciega en Test...")
    final_eval()

    print("\n Pipeline completo ejecutado exitosamente.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestador TRPII Fraude")
    parser.add_argument(
        "--all", action="store_true", help="Ejecuta todo el pipeline de principio a fin"
    )
    args = parser.parse_args()

    if args.all:
        run_all()
    else:
        print("Modo por defecto: preparando datos. Usa '--all' para correr el pipeline completo.")
        prepare_data()


if __name__ == "__main__":
    main()
