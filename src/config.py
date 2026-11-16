"""Configuracion centralizada del proyecto: rutas, semillas, costos y columnas."""

from pathlib import Path

# Rutas
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "fraud_dataset_v2.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
OOF_DIR = PROCESSED_DIR / "oof"

# Semillas
RANDOM_STATE = 42
# Semilla del split 75/25, independiente del resto para poder cambiarla sola.
SPLIT_RANDOM_STATE = 42

# Particion
TEST_SIZE = 0.25
N_FOLDS = 5

# Modulo de anomalias (Tarea 5): split temporal train / calibracion / test
TEMPORAL_FRACTIONS = (0.6, 0.2, 0.2)
ANOMALY_DIR = PROCESSED_DIR / "anomalies"
# Proporcion de transacciones que se marcan como sospechosas. Se fija a priori
# (no usa `fraude`); es un supuesto de capacidad de revision, modificable.
ANOMALY_ALERT_RATE = 0.02

# Columnas
ID_COLUMN = "id"
TARGET_COLUMN = "fraude"
DATE_COLUMN = "fecha"

# Columnas que nunca entran como entrada del modelo (ver features.md).
# `id` y `fraude` se conservan en los DataFrames pero no son features.
NON_MODEL_COLUMNS = [ID_COLUMN, TARGET_COLUMN]
DROPPED_COLUMNS = [
    "k",  # identificador unico por fila
    "r",  # redundante con `c`
    "d",  # correlacion > 0.99 con `q`
    "p",  # biyectiva con `s` (Y <-> 100, N <-> 0)
    "e",  # 44% de ceros y correlacion ~0 con el resto
]

# Features derivadas de `score`: excluir (junto con `score`) del modulo de anomalias.
SCORE_DERIVED_FEATURES = ["score_bin", "score_x_logmonto", "j_mean_score", "score_rel_j"]

# Columnas de entrada de los pipelines (despues de add_basic_features y
# drop_unused_columns). `c`, `f`, `m` y `l` entran en version logaritmica.
NUMERIC_FEATURES = [
    "a", "b", "h", "n", "q", "s", "score",
    "log_monto", "log_f", "log_c", "log_m", "log_l",
    "f_neg", "q_neg",
    "hora", "hora_sin", "hora_cos", "dow_sin", "dow_cos",
    "es_fin_de_semana", "es_madrugada",
    "score_x_logmonto",
    "len_titulo", "n_palabras", "cant_digitos", "pct_mayusculas", "cant_simbolos",
]
CATEGORICAL_FEATURES = ["o", "hora_bin", "score_bin"]
TARGET_ENCODED_FEATURES = ["j"]
# Target encoding adicional (perfil de comportamiento); se suma a `j`.
BEHAVIOR_TARGET_EXTRA = ["g"]
FREQUENCY_FEATURES = ("j", "g")
RARE_GROUPED_FEATURE = "g"
# Columnas que necesita GroupAmountFeatures (no son salidas del modelo por si solas).
GROUP_AMOUNT_INPUTS = ["monto", "log_monto", "hora", "g", "j"]
MIN_CATEGORY_FREQUENCY = 100

# Filas validas: las ultimas 100.000 del archivo original son sinteticas y
# tienen id duplicado, por lo que quedan 150.000 tras eliminar duplicados.
EXPECTED_ROWS = 150_000

# Costos oficiales de negocio (PDF de la consigna)
COSTO_FP = 5.0
COSTO_FN = 200.0

# Maximo de falsos positivos tolerables en el conjunto de train (112.500 filas).
# Justificacion de negocio: en ~42 dias de operacion (6 semanas), un equipo
# de 4 analistas revisando ~12-13 casos manuales diarios c/u puede absorber ~50 FP/dia,
# lo que equivale a ~2.100 FP en el periodo de train (tasa de error ~2.0% sobre legitimos).
# En test (37.500 filas), la capacidad proporcional equivale a ~712 FP.
FP_MAX_RATE = 0.02
FP_MAX = 2137
