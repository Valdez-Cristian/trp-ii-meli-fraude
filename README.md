# Detección de Fraude en Transacciones Financieras (Mercado Libre)
**Materia:** Taller de Resolución de Problemas II (TRPII) — Universidad Austral  
**Alumnos:** Valdez Cristian, Rivas Francisco, Villacorta Martina
**Docente:** Andrés D'Ambrosio  

---

## 1. Visión General del Proyecto
Este proyecto implementa una solución integral de Machine Learning para la detección de fraude financiero en transacciones de comercio electrónico (Mercado Libre), optimizando directamente el **impacto financiero del negocio** y garantizando viabilidad operativa para los analistas de prevención de fraude.

### Matriz de Costo del Negocio:
$$\text{Costo Total} = (\text{FP} \times \$5) + (\text{FN} \times \$200)$$
* **Falso Positivo (FP):** Fricción, contacto manual con cliente legítimo (\$5 c/u).
* **Falso Negativo (FN):** Pérdida directa por fraude y contracargo (\$200 c/u). Ratio de penalización 40:1.
* **Restricción de Capacidad:** Límite operativo fijado en $\text{FP}_{MAX} = 2,137$ transacciones (~50 alertas diarias absorbibles por un equipo de 4 analistas durante 6 semanas).

---

## 2. Resultados Oficiales de Test (Evaluación Ciega, 25% de la muestra)

| Métrica | Score Legado | Modelo Propuesto (LightGBM Tuned) | Impacto / Mejora |
|---|---|---|---|
| **PR-AUC** | ~0.150 | **0.4750** | Triplica el poder discriminativo |
| **ROC-AUC** | 0.680 | **0.8905** | +21 puntos de área |
| **Precisión** | 28.0% | **51.11%** | 1 de cada 2 alertas es fraude real |
| **Recall (Detección)** | 11.8% (222 fraudes) | **42.93% (805 fraudes)** | **+3.6x fraudes detectados** |
| **Falsos Positivos** | 572 (1.6%) | 770 (2.16%) | Operación controlada y viable |
| **Costo Total Financiero** | **$333,635** | **$217,850** | **-$115,785 de pérdida evitada** |
| **Ahorro Neto** | Base | **34.7% de ahorro financiero** | Frente a no hacer nada: **41.9% ($157k)** |

---

## 3. Estructura del Código

```text
dev/
├── data/
│   ├── fraud_dataset_v2.csv          # Dataset original (250.000 filas)
│   └── processed/                    # Derivados procesados
│       ├── train.parquet, test.parquet
│       ├── oof/                      # Predicciones OOF y umbrales óptimos
│       ├── anomalies/                # Resultados de Tarea 5 (Isolation Forest, LOF, PCA)
│       ├── explain/                  # Importancias globales y explicaciones SHAP
│       ├── sensitivity/              # Tablas de barrido de costos y caso de negocio
│       └── test_evaluation.json      # Evaluación ciega oficial en Test
├── src/
│   ├── config.py                     # Rutas, costos ($5/$200), semillas y FP_MAX
│   ├── data.py                       # Ingesta Polars, depuración de filas sintéticas y splits
│   ├── features.py                   # Features sin estado y transformadores Scikit-Learn
│   ├── pipelines.py                  # Ensamblado de ColumnTransformer
│   ├── models.py                     # Catálogo de estimadores (Logistic, RF, LightGBM)
│   ├── metrics.py                    # Matriz de confusión, PR-AUC y función de costo
│   ├── optimization.py               # Selección de umbrales OOF con restricción FP_MAX
│   ├── explain.py                    # TreeSHAP nativo de LightGBM y explicaciones locales
│   └── anomalies.py                  # Detectores no supervisados (Tarea 5)
├── scripts/
│   ├── prepare_data.py               # Ingesta y deduplicado -> train/test
│   ├── train_models.py               # CV OOF de modelos
│   ├── tune_models.py                # Optimización de hiperparámetros
│   ├── recompute_thresholds.py       # Reoptimización de umbrales bajo FP_MAX
│   ├── sensitivity_analysis.py       # Barrido de sensibilidad de costos
│   ├── explain_model.py              # Cálculo de importancias y SHAP
│   ├── detect_anomalies.py           # Detección no supervisada (Tarea 5)
│   └── final_eval.py                 # Evaluación ciega final sobre test.parquet
└── pyproject.toml                    # Configuración del paquete y dependencias (uv)
```

---

## 4. Cómo Reproducir la Solución

Desde la carpeta `dev/`:

```bash
# 1. Preparar datos (deduplicar y generar train/test)
uv run python -m scripts.prepare_data

# 2. Entrenar y obtener probabilidades out-of-fold
uv run python -m scripts.train_models logistic

# 3. Optimizar umbrales bajo capacidad operativa (FP_MAX)
uv run python -m scripts.recompute_thresholds

# 4. Generar interpretabilidad SHAP y análisis de sensibilidad
uv run python -m scripts.explain_model
uv run python -m scripts.sensitivity_analysis

# 5. Ejecutar detección de anomalías no supervisadas (Tarea 5)
uv run python -m scripts.detect_anomalies

# 6. Evaluación ciega final sobre el conjunto de Test
uv run python -m scripts.final_eval # modificar modelo en linea 31 de scripts/final_eval.py
```
