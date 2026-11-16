# ==============================================================================
# TRABAJO PRACTICO INTEGRADOR: DETECCION DE FRAUDE
# MODELADO, OPTIMIZACION DE UMBRALES Y COSTOS DE NEGOCIO
# =============================================================================
#
# Requisito: colocar "fraud_dataset_v2.csv" en el directorio de trabajo de R
# o modificar la ruta indicada en la seccion de carga.
# ==============================================================================


# ==============================================================================
# 1. LIBRERIAS Y CARGA DE DATOS
# ==============================================================================

library(tidyverse)
library(lubridate)
library(caret)
library(PRROC)
library(randomForest)
library(xgboost)

# Si el archivo no esta en el directorio de trabajo, reemplazar por su ruta.
df_raw <- read.csv("../data/fraud_dataset_v2.csv")

# primeras 150.000 transacciones.
df_clean <- df_raw %>%
  slice(1:150000)

# Eliminacion de variables clonadas, ruido e indice
cols_a_remover <- c("Unnamed: 0", "r", "q", "s", "k", "e")
df_clean <- df_clean %>%
  select(-any_of(cols_a_remover))

# Tratamiento de valores faltantes y valores -1.
# En la variable 'o', la ausencia de informacion se conserva como categoria.
df_clean <- df_clean %>%
  mutate(
    across(where(is.numeric), ~ na_if(., -1)),
    o_estado = if_else(is.na(o), "Nulo", as.character(o))
  )


# ==============================================================================
# 2. VARIABLES NECESARIAS PARA LOS MODELOS
# ==============================================================================

df_features <- df_clean %>%
  mutate(
    fecha_dt = ymd_hms(fecha),
    hora = hour(fecha_dt),
    dia_semana = wday(fecha_dt, week_start = 1),
    es_fin_de_semana = if_else(dia_semana >= 6, 1, 0),
    es_madrugada = if_else(hora >= 1 & hora <= 4, 1, 0),
    es_nulo_o = if_else(o_estado == "Nulo", 1, 0),
    monto_log = log1p(monto),

    # Grado de incompletitud de los campos considerados relevantes.
    grado_orfandad = rowSums(
      is.na(across(any_of(c("b", "c", "d", "f", "g", "l", "m", "o"))))
    )
  )

# Se mantiene la variable como factor para que sus niveles sean consistentes
# entre entrenamiento, validacion y test.
df_features$o_estado <- factor(df_features$o_estado)


# ==============================================================================
# 3. PARTICION TRAIN / VALIDATION / TEST
# ==============================================================================

# Se conserva la particion original del preprocesamiento:
# 70% para desarrollo y 30% para test.
set.seed(42)
train_Index <- createDataPartition(
  df_features$fraude,
  p = 0.7,
  list = FALSE
)

train_set <- df_features[train_Index, ]
test_set  <- df_features[-train_Index, ]

# Dentro del 70% de desarrollo, se separa 80% para entrenamiento
# y 20% para validacion. En terminos del total: 56% train, 14% validation,
# 30% test.
set.seed(42)
idx_train_val <- createDataPartition(
  train_set$fraude,
  p = 0.8,
  list = FALSE
)

df_train_val <- train_set[idx_train_val, ]
df_val       <- train_set[-idx_train_val, ]
df_test      <- test_set

cat("Dimensiones de entrenamiento:", dim(df_train_val), "\n")
cat("Dimensiones de validacion:", dim(df_val), "\n")
cat("Dimensiones de test:", dim(df_test), "\n")


# ==============================================================================
# 4. FUNCIONES AUXILIARES: COSTOS, UMBRALES Y EVALUACION
# ==============================================================================

costo_fp <- 5
costo_fn <- 200

calcular_costo_total <- function(obs, probs, umbral, c_fp, c_fn) {
  preds <- ifelse(probs >= umbral, 1, 0)

  mat <- table(
    Prediccion = factor(preds, levels = c(0, 1)),
    Real = factor(obs, levels = c(0, 1))
  )

  fp <- as.numeric(mat[2, 1])
  fn <- as.numeric(mat[1, 2])

  (fp * c_fp) + (fn * c_fn)
}

obtener_umbral_optimo <- function(probs, reales) {
  umbrales <- seq(0.01, 0.99, by = 0.01)

  costos <- sapply(umbrales, function(u) {
    calcular_costo_total(reales, probs, u, costo_fp, costo_fn)
  })

  umbrales[which.min(costos)]
}

obtener_umbral_optimo_fino <- function(probs, reales) {
  umbrales <- seq(0.005, 0.50, by = 0.005)

  costos <- sapply(umbrales, function(u) {
    preds <- ifelse(probs >= u, 1, 0)
    fp <- sum(preds == 1 & reales == 0)
    fn <- sum(preds == 0 & reales == 1)

    (fp * costo_fp) + (fn * costo_fn)
  })

  umbrales[which.min(costos)]
}

calcular_auc_pr <- function(reales, probs) {
  fg <- probs[reales == 1]
  bg <- probs[reales == 0]

  pr.curve(
    scores.class0 = fg,
    scores.class1 = bg
  )$auc.integral
}

evaluar_modelo_test <- function(nombre_modelo, probs_test, y_real,
                                umbral, costo_fp = 5, costo_fn = 200) {
  preds_clase <- ifelse(probs_test >= umbral, 1, 0)

  mat <- table(
    Prediccion = factor(preds_clase, levels = c(0, 1)),
    Real = factor(y_real, levels = c(0, 1))
  )

  fp <- as.numeric(mat[2, 1])
  fn <- as.numeric(mat[1, 2])
  tp <- as.numeric(mat[2, 2])

  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  auc_pr <- calcular_auc_pr(y_real, probs_test)
  costo_total <- (fp * costo_fp) + (fn * costo_fn)

  data.frame(
    Modelo = nombre_modelo,
    Umbral = umbral,
    AUC_PR = round(auc_pr, 4),
    Precision = round(precision, 4),
    Falsos_Positivos = fp,
    Falsos_Negativos = fn,
    Costo_Total = costo_total
  )
}


# ==============================================================================
# 5. MODELO BASE: REGRESION LOGISTICA
# ==============================================================================

formula_modelo <- fraude ~ monto_log + hora + es_madrugada +
  es_fin_de_semana + grado_orfandad + score + o_estado

modelo_base <- glm(
  formula_modelo,
  data = df_train_val,
  family = binomial
)

pred_val_prob <- predict(
  modelo_base,
  newdata = df_val,
  type = "response"
)

auc_pr_base_val <- calcular_auc_pr(df_val$fraude, pred_val_prob)
cat("\nAUC-PR Regresion Logistica en validacion:", auc_pr_base_val, "\n")

# Optimizacion del umbral con costos de negocio.
umbrales <- seq(0.01, 0.99, by = 0.01)
costos <- sapply(
  umbrales,
  function(u) calcular_costo_total(
    df_val$fraude, pred_val_prob, u, costo_fp, costo_fn
  )
)

umbral_optimo <- umbrales[which.min(costos)]
cat("Umbral optimo Regresion Logistica:", umbral_optimo, "\n")

# Evaluacion final en test.
pred_test_prob <- predict(
  modelo_base,
  newdata = df_test,
  type = "response"
)

mat_test <- table(
  Prediccion = factor(ifelse(pred_test_prob >= umbral_optimo, 1, 0),
                      levels = c(0, 1)),
  Real = factor(df_test$fraude, levels = c(0, 1))
)

cat("\nMatriz de confusion - Regresion Logistica:\n")
print(mat_test)

fp_test <- as.numeric(mat_test[2, 1])
fn_test <- as.numeric(mat_test[1, 2])
costo_final_test <- (fp_test * costo_fp) + (fn_test * costo_fn)

cat("FP:", fp_test, "| FN:", fn_test, "\n")
cat("Costo operativo en test: $", costo_final_test, "\n")


# ==============================================================================
# 6. RANDOM FOREST BASE
# ==============================================================================

# La version recibida referenciaba df_train_val_bag, df_val_bag y df_test_bag,
# pero esos objetos no se construian en ninguno de los tres archivos.
# Para que el modelo base pueda ejecutarse sin inventar un procedimiento de
# bagging no documentado, se entrena sobre las particiones originales.
df_train_val_rf <- df_train_val %>%
  mutate(fraude_f = factor(if_else(fraude == 1, "Si", "No"),
                           levels = c("No", "Si")))

df_val_rf <- df_val %>%
  mutate(fraude_f = factor(if_else(fraude == 1, "Si", "No"),
                           levels = c("No", "Si")))

df_test_rf <- df_test %>%
  mutate(fraude_f = factor(if_else(fraude == 1, "Si", "No"),
                           levels = c("No", "Si")))

ctrl_cv <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

set.seed(42)
modelo_rf <- train(
  fraude_f ~ monto_log + hora + es_madrugada + es_fin_de_semana +
    grado_orfandad + score + o_estado,
  data = df_train_val_rf,
  method = "rf",
  trControl = ctrl_cv,
  ntree = 100,
  tuneGrid = expand.grid(mtry = c(2, 3, 4))
)

print(modelo_rf)

pred_val_rf_prob <- predict(
  modelo_rf,
  newdata = df_val_rf,
  type = "prob"
)[, "Si"]

auc_pr_rf <- calcular_auc_pr(df_val_rf$fraude, pred_val_rf_prob)
cat("AUC-PR Random Forest Base en validacion:", auc_pr_rf, "\n")

costos_rf <- sapply(umbrales, function(u) {
  calcular_costo_total(
    df_val_rf$fraude, pred_val_rf_prob, u, costo_fp, costo_fn
  )
})

umbral_optimo_rf <- umbrales[which.min(costos_rf)]
cat("Umbral optimo Random Forest Base:", umbral_optimo_rf, "\n")

pred_test_rf_prob <- predict(
  modelo_rf,
  newdata = df_test_rf,
  type = "prob"
)[, "Si"]

mat_test_rf <- table(
  Prediccion = factor(ifelse(pred_test_rf_prob >= umbral_optimo_rf, 1, 0),
                      levels = c(0, 1)),
  Real = factor(df_test_rf$fraude, levels = c(0, 1))
)

cat("\nMatriz de confusion - Random Forest Base:\n")
print(mat_test_rf)

fp_test_rf <- as.numeric(mat_test_rf[2, 1])
fn_test_rf <- as.numeric(mat_test_rf[1, 2])
costo_final_rf <- (fp_test_rf * costo_fp) + (fn_test_rf * costo_fn)

cat("FP:", fp_test_rf, "| FN:", fn_test_rf, "\n")
cat("Costo operativo Random Forest Base en test: $", costo_final_rf, "\n")


# ==============================================================================
# 7. FEATURE ENGINEERING PARA RANDOM FOREST Y XGBOOST
# ==============================================================================

enriquecer_datos <- function(df) {
  df %>%
    mutate(
      pais_grupo = factor(case_when(
        g == "BR" ~ "Brasil",
        g == "AR" ~ "Argentina",
        TRUE ~ "Otros"
      )),
      score_decil = ntile(score, 10),
      interaccion_riesgo_madrugada = factor(
        interaction(a, es_madrugada, drop = TRUE)
      ),
      fraude_f = factor(
        if_else(fraude == 1, "Si", "No"),
        levels = c("No", "Si")
      )
    )
}

# Se fijan los niveles categoricos con el conjunto completo de desarrollo
# para evitar incompatibilidades entre train, validacion y test al predecir.
df_features_fe <- enriquecer_datos(df_features)

df_train_val_fe <- enriquecer_datos(df_train_val)
df_val_fe       <- enriquecer_datos(df_val)
df_test_fe      <- enriquecer_datos(df_test)

niveles_pais <- levels(df_features_fe$pais_grupo)
niveles_interaccion <- levels(df_features_fe$interaccion_riesgo_madrugada)

df_train_val_fe <- df_train_val_fe %>%
  mutate(
    pais_grupo = factor(pais_grupo, levels = niveles_pais),
    interaccion_riesgo_madrugada = factor(
      interaccion_riesgo_madrugada,
      levels = niveles_interaccion
    )
  )

df_val_fe <- df_val_fe %>%
  mutate(
    pais_grupo = factor(pais_grupo, levels = niveles_pais),
    interaccion_riesgo_madrugada = factor(
      interaccion_riesgo_madrugada,
      levels = niveles_interaccion
    )
  )

df_test_fe <- df_test_fe %>%
  mutate(
    pais_grupo = factor(pais_grupo, levels = niveles_pais),
    interaccion_riesgo_madrugada = factor(
      interaccion_riesgo_madrugada,
      levels = niveles_interaccion
    )
  )

formula_rf_completa <- fraude_f ~ monto_log + hora + es_madrugada +
  es_fin_de_semana + grado_orfandad + score + score_decil +
  o_estado + pais_grupo + interaccion_riesgo_madrugada

ctrl_cv <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)


# ==============================================================================
# 9. RANDOM FOREST CON FEATURE ENGINEERING + UNDERSAMPLING
# ==============================================================================

ctrl_cv_down <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "down"
)

set.seed(42)
modelo_rf_undersample <- train(
  formula_rf_completa,
  data = df_train_val_fe,
  method = "rf",
  trControl = ctrl_cv_down,
  ntree = 100,
  tuneGrid = expand.grid(mtry = c(3, 4, 5))
)

cat("\n--- RESULTADOS RANDOM FOREST (CON UNDERSAMPLING) ---\n")
print(modelo_rf_undersample)

pred_val_under <- predict(
  modelo_rf_undersample,
  newdata = df_val_fe,
  type = "prob"
)[, "Si"]

auc_pr_under <- calcular_auc_pr(df_val_fe$fraude, pred_val_under)
cat("AUC-PR RF + Undersampling:", auc_pr_under, "\n")

costos_under <- sapply(umbrales, function(u) {
  calcular_costo_total(
    df_val_fe$fraude, pred_val_under, u, costo_fp, costo_fn
  )
})

u_rf_under <- umbrales[which.min(costos_under)]
cat("Umbral optimo RF + Undersampling:", u_rf_under, "\n")

pred_test_under_prob <- predict(
  modelo_rf_undersample,
  newdata = df_test_fe,
  type = "prob"
)[, "Si"]

mat_test_under <- table(
  Prediccion = factor(ifelse(pred_test_under_prob >= u_rf_under, 1, 0),
                      levels = c(0, 1)),
  Real = factor(df_test_fe$fraude, levels = c(0, 1))
)

cat("\nMatriz de confusion - RF + Undersampling:\n")
print(mat_test_under)

fp_test_u <- as.numeric(mat_test_under[2, 1])
fn_test_u <- as.numeric(mat_test_under[1, 2])
costo_u <- (fp_test_u * costo_fp) + (fn_test_u * costo_fn)

cat("FP:", fp_test_u, "| FN:", fn_test_u, "\n")
cat("Costo operativo RF + Undersampling en test: $", costo_u, "\n")


# ==============================================================================
# 10. XGBOOST (GRADIENT BOOSTING)
# ==============================================================================

vars_xgb <- c(
  "monto_log", "hora", "es_madrugada", "es_fin_de_semana",
  "grado_orfandad", "score", "score_decil", "o_estado",
  "pais_grupo", "interaccion_riesgo_madrugada"
)

# Se construyen las matrices por separado y luego se alinean sus columnas.
# Esto evita diferencias de dimensiones cuando una categoria aparece solo
# en alguna de las particiones.
options(na.action = "na.pass")

X_train <- model.matrix(~ . - 1, data = df_train_val_fe[, vars_xgb])
X_val   <- model.matrix(~ . - 1, data = df_val_fe[, vars_xgb])
X_test  <- model.matrix(~ . - 1, data = df_test_fe[, vars_xgb])

columnas_xgb <- Reduce(
  union,
  list(colnames(X_train), colnames(X_val), colnames(X_test))
)

alinear_columnas <- function(matriz, columnas) {
  faltantes <- setdiff(columnas, colnames(matriz))

  if (length(faltantes) > 0) {
    matriz <- cbind(
      matriz,
      matrix(
        0,
        nrow = nrow(matriz),
        ncol = length(faltantes),
        dimnames = list(NULL, faltantes)
      )
    )
  }

  matriz[, columnas, drop = FALSE]
}

X_train <- alinear_columnas(X_train, columnas_xgb)
X_val   <- alinear_columnas(X_val, columnas_xgb)
X_test  <- alinear_columnas(X_test, columnas_xgb)

options(na.action = "na.omit")

y_train <- df_train_val_fe$fraude
y_val   <- df_val_fe$fraude
y_test  <- df_test_fe$fraude

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dval   <- xgb.DMatrix(data = X_val, label = y_val)
dtest  <- xgb.DMatrix(data = X_test, label = y_test)

params_xgb <- list(
  objective = "binary:logistic",
  eval_metric = "aucpr",
  max_depth = 4,
  eta = 0.05,
  min_child_weight = 50,
  subsample = 0.7,
  colsample_bytree = 0.8
)

set.seed(42)
modelo_xgb <- xgb.train(
  params = params_xgb,
  data = dtrain,
  nrounds = 1000,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 30,
  print_every_n = 50
)

pred_val_xgb <- predict(modelo_xgb, dval)
auc_pr_xgb_val <- calcular_auc_pr(y_val, pred_val_xgb)
cat("\nAUC-PR XGBoost en validacion:", round(auc_pr_xgb_val, 4), "\n")

u_xgb <- obtener_umbral_optimo_fino(pred_val_xgb, y_val)
cat("Umbral optimo XGBoost en validacion:", u_xgb, "\n")

pred_test_xgb_prob <- predict(modelo_xgb, dtest)
pred_test_xgb_clase <- ifelse(pred_test_xgb_prob >= u_xgb, 1, 0)

mat_test_xgb <- table(
  Prediccion = factor(pred_test_xgb_clase, levels = c(0, 1)),
  Real = factor(y_test, levels = c(0, 1))
)

cat("\nMatriz de confusion - XGBoost:\n")
print(mat_test_xgb)

fp_test_xgb <- as.numeric(mat_test_xgb[2, 1])
fn_test_xgb <- as.numeric(mat_test_xgb[1, 2])
costo_total_xgb <- (fp_test_xgb * costo_fp) + (fn_test_xgb * costo_fn)
auc_pr_xgb_test <- calcular_auc_pr(y_test, pred_test_xgb_prob)

cat("\n========================================\n")
cat("RESULTADOS FINALES EN TEST (XGBOOST + FE)\n")
cat("========================================\n")
cat("AUC-PR en Test:       ", round(auc_pr_xgb_test, 4), "\n")
cat("Falsos Positivos (FP):", fp_test_xgb, "\n")
cat("Falsos Negativos (FN):", fn_test_xgb, "\n")
cat("Costo Operativo Total: $", costo_total_xgb, "\n")
cat("========================================\n")


# ==============================================================================
# 11. STACKING: XGBOOST + REGRESION LOGISTICA
# ==============================================================================

formula_logistica <- fraude_f ~ monto_log + hora + es_madrugada +
  es_fin_de_semana + grado_orfandad + score + score_decil +
  o_estado + pais_grupo + interaccion_riesgo_madrugada

modelo_log_meta <- glm(
  formula_logistica,
  data = df_train_val_fe,
  family = binomial
)

pred_val_log <- predict(
  modelo_log_meta,
  newdata = df_val_fe,
  type = "response"
)

pred_test_log <- predict(
  modelo_log_meta,
  newdata = df_test_fe,
  type = "response"
)

df_meta_train <- data.frame(
  prob_xgb = pred_val_xgb,
  prob_log = pred_val_log,
  fraude = df_val_fe$fraude
)

df_meta_test <- data.frame(
  prob_xgb = pred_test_xgb_prob,
  prob_log = pred_test_log,
  fraude = df_test_fe$fraude
)

meta_modelo <- glm(
  fraude ~ prob_xgb + prob_log,
  data = df_meta_train,
  family = binomial
)

cat("\n--- COEFICIENTES DEL META-MODELO (XGB + LOGISTICA) ---\n")
print(summary(meta_modelo))

pred_val_stack <- predict(
  meta_modelo,
  newdata = df_meta_train,
  type = "response"
)

pred_test_stack <- predict(
  meta_modelo,
  newdata = df_meta_test,
  type = "response"
)

auc_pr_stack_val <- calcular_auc_pr(df_meta_train$fraude, pred_val_stack)
cat("AUC-PR Stacking en validacion:", round(auc_pr_stack_val, 4), "\n")

u_stack <- obtener_umbral_optimo_fino(
  pred_val_stack,
  df_meta_train$fraude
)
cat("Umbral optimo Stacking:", u_stack, "\n")

pred_test_stack_clase <- ifelse(pred_test_stack >= u_stack, 1, 0)

mat_test_stack <- table(
  Prediccion = factor(pred_test_stack_clase, levels = c(0, 1)),
  Real = factor(df_meta_test$fraude, levels = c(0, 1))
)

cat("\nMatriz de confusion - Stacking (XGB + Logistica):\n")
print(mat_test_stack)

fp_test_stack <- as.numeric(mat_test_stack[2, 1])
fn_test_stack <- as.numeric(mat_test_stack[1, 2])
costo_total_stack <- (fp_test_stack * costo_fp) + (fn_test_stack * costo_fn)
auc_pr_stack_test <- calcular_auc_pr(df_meta_test$fraude, pred_test_stack)

cat("AUC-PR Stacking en test:", round(auc_pr_stack_test, 4), "\n")
cat("FP:", fp_test_stack, "| FN:", fn_test_stack, "\n")
cat("Costo operativo Stacking en test: $", costo_total_stack, "\n")

# ==============================================================================
# 13. XGBOOST CON UMBRAL FORZADO
# ==============================================================================

umbral_forzado <- 0.15

pred_test_forzado_clase <- ifelse(
  pred_test_xgb_prob >= umbral_forzado, 1, 0
)

mat_test_forzado <- table(
  Prediccion = factor(pred_test_forzado_clase, levels = c(0, 1)),
  Real = factor(y_test, levels = c(0, 1))
)

cat("\nMatriz de confusion XGBoost con umbral forzado =", umbral_forzado, "\n")
print(mat_test_forzado)

fp_forzado <- as.numeric(mat_test_forzado[2, 1])
fn_forzado <- as.numeric(mat_test_forzado[1, 2])
tp_forzado <- as.numeric(mat_test_forzado[2, 2])

precision_forzada <- ifelse(
  (tp_forzado + fp_forzado) > 0,
  tp_forzado / (tp_forzado + fp_forzado),
  0
)

costo_total_forzado <- (fp_forzado * costo_fp) + (fn_forzado * costo_fn)

cat("\n========================================\n")
cat("RESULTADOS EN TEST (UMBRAL FORZADO:", umbral_forzado, ")\n")
cat("========================================\n")
cat("AUC-PR en Test:       ", round(auc_pr_xgb_test, 4), "\n")
cat("Falsos Positivos (FP):", fp_forzado, "\n")
cat("Falsos Negativos (FN):", fn_forzado, "\n")
cat("Precision:            ", round(precision_forzada, 4), "\n")
cat("Costo Operativo Total: $", costo_total_forzado, "\n")
cat("========================================\n")


# ==============================================================================
# 14. TABLA COMPARATIVA FINAL
# ==============================================================================

res_log_base <- evaluar_modelo_test(
  "Regresion Logistica Base",
  pred_test_prob, df_test$fraude, umbral_optimo
)

res_rf_base <- evaluar_modelo_test(
  "Random Forest Base",
  pred_test_rf_prob, df_test_rf$fraude, umbral_optimo_rf
)


res_rf_under <- evaluar_modelo_test(
  "RF + FE + Undersampling",
  pred_test_under_prob, df_test_fe$fraude, u_rf_under
)

res_xgb <- evaluar_modelo_test(
  "XGBoost + FE",
  pred_test_xgb_prob, y_test, u_xgb
)


res_xgb_forzado <- data.frame(
  Modelo = paste0("XGBoost (Umbral Forzado = ", umbral_forzado, ")"),
  Umbral = umbral_forzado,
  AUC_PR = round(auc_pr_xgb_test, 4),
  Precision = round(precision_forzada, 4),
  Falsos_Positivos = fp_forzado,
  Falsos_Negativos = fn_forzado,
  Costo_Total = costo_total_forzado
)

tabla_comparativa_final <- bind_rows(
  res_log_base,
  res_rf_base,
  res_rf_under,
  res_xgb,
  res_stack_log,
  res_xgb_forzado
) %>%
  arrange(Costo_Total)

cat("\n==========================================================================\n")
cat("TABLA COMPARATIVA FINAL DE MODELOS (ORDENADA POR COSTO OPERATIVO)\n")
cat("==========================================================================\n")
print(tabla_comparativa_final)
