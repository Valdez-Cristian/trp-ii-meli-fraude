paquetes_requeridos <- c("tidyverse", "lubridate", "caret", "PRROC", "ROSE",
                         "rpart", "rpart.plot", "glmnet", "ranger", "xgboost")
paquetes_faltantes  <- paquetes_requeridos[!(paquetes_requeridos %in% installed.packages()[,"Package"])]
if(length(paquetes_faltantes) > 0) install.packages(paquetes_faltantes, dependencies = TRUE)

library(xgboost)
library(tidyverse)
library(lubridate)
library(PRROC)
library(caret)
library(ROSE)
library(rpart)
library(rpart.plot)
library(glmnet)
library(ranger)


ruta_datos <- "./../data/fraud_dataset_v2.csv"
if (!file.exists(ruta_datos)) {
  stop("No se encontró el archivo: ", normalizePath(ruta_datos, mustWork = FALSE),
       "\nRevisá el directorio de trabajo.")
}

# 'fecha' se lee como texto porque el script la parsea después con ymd_hms()
df_raw <- read_csv(ruta_datos, col_types = cols(fecha = col_character()),
                   show_col_types = FALSE) %>%
  mutate(fraude = as.integer(fraude))   # el script espera la variable objetivo en 0/1

# Split 60% / 20% / 20% (train / validation / test)
df_raw <- df_raw %>% arrange(ymd_hms(fecha))
n      <- nrow(df_raw)
i_tr   <- floor(0.60 * n)
i_val  <- floor(0.80 * n)

df_train_val <- df_raw[1:i_tr, ]               # entrenamiento (stats de encoding + modelos)
df_val       <- df_raw[(i_tr + 1):i_val, ]     # validación (elige el umbral óptimo)
test_set     <- df_raw[(i_val + 1):n, ]        # test (evaluación final)

cat("Train:", nrow(df_train_val), "| Val:", nrow(df_val), "| Test:", nrow(test_set), "\n")
cat("Tasa de fraude -> Train:", round(mean(df_train_val$fraude), 4),
    "| Val:", round(mean(df_val$fraude), 4),
    "| Test:", round(mean(test_set$fraude), 4), "\n")

# ------------------------------------------------------------------------------
# 1. FUNCIONES DE EVALUACIÓN (UMBRAL ÓPTIMO LIBRE EN VALIDATION -> TEST)
# ------------------------------------------------------------------------------
evaluar_modelo <- function(nombre, probs, y_real, c_fp = 5, c_fn = 200) {
  y_num <- as.numeric(as.character(y_real))

  fg <- probs[y_num == 1]
  bg <- probs[y_num == 0]
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = FALSE)
  auc_pr_val <- pr$auc.integral

  umbrales <- seq(0.01, 0.99, by = 0.01)
  costos <- sapply(umbrales, function(u) {
    preds <- ifelse(probs >= u, 1, 0)
    fp <- sum(preds == 1 & y_num == 0)
    fn <- sum(preds == 0 & y_num == 1)
    (fp * c_fp) + (fn * c_fn)
  })

  idx_opt   <- which.min(costos)
  u_opt     <- umbrales[idx_opt]
  preds_opt <- ifelse(probs >= u_opt, 1, 0)

  fp_opt <- sum(preds_opt == 1 & y_num == 0)
  fn_opt <- sum(preds_opt == 0 & y_num == 1)
  tp_opt <- sum(preds_opt == 1 & y_num == 1)

  data.frame(
    Modelo        = nombre,
    AUC_PR        = round(auc_pr_val, 4),
    Umbral_Optimo = u_opt,
    Precision     = round(tp_opt / (tp_opt + fp_opt), 4),
    Recall        = round(tp_opt / (tp_opt + fn_opt), 4),
    FP            = fp_opt,
    FN            = fn_opt,
    Costo_Total   = costos[idx_opt]
  )
}

evaluar_en_test <- function(nombre, probs_test, y_real_test, umbral_val, c_fp = 5, c_fn = 200) {
  y_num <- as.numeric(as.character(y_real_test))

  fg <- probs_test[y_num == 1]
  bg <- probs_test[y_num == 0]
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = FALSE)
  auc_pr_val <- pr$auc.integral

  preds_test <- ifelse(probs_test >= umbral_val, 1, 0)

  fp_test <- sum(preds_test == 1 & y_num == 0)
  fn_test <- sum(preds_test == 0 & y_num == 1)
  tp_test <- sum(preds_test == 1 & y_num == 1)

  data.frame(
    Modelo           = nombre,
    AUC_PR_Test      = round(auc_pr_val, 4),
    Umbral_Val       = umbral_val,
    Precision_Test   = round(tp_test / (tp_test + fp_test), 4),
    Recall_Test      = round(tp_test / (tp_test + fn_test), 4),
    FP_Test          = fp_test,
    FN_Test          = fn_test,
    Costo_Total_Test = (fp_test * c_fp) + (fn_test * c_fn)
  )
}

obtener_umbral <- function(tabla_val, nombre_mod) {
  tabla_val$Umbral_Optimo[tabla_val$Modelo == nombre_mod]
}

# ------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING UNIFICADO (SIN DATA LEAKAGE NI COLISIÓN DE NOMBRES)
# ------------------------------------------------------------------------------
global_mean_fraude <- mean(df_train_val$fraude)
monto_prom_global  <- mean(df_train_val$monto, na.rm = TRUE)

stats_cat_a <- df_train_val %>%
  filter(!is.na(a)) %>%
  group_by(a) %>%
  summarise(
    cnt_a          = n(),
    fraudes_a      = sum(fraude),
    target_enc_a   = (fraudes_a + 10 * global_mean_fraude) / (cnt_a + 10),
    monto_prom_cat = mean(monto, na.rm = TRUE),
    .groups        = "drop"
  ) %>%
  select(a, target_enc_a, monto_prom_cat)

stats_cat_j <- df_train_val %>%
  filter(!is.na(j)) %>%
  group_by(j) %>%
  summarise(
    cnt_j        = n(),
    fraudes_j    = sum(fraude),
    target_enc_j = (fraudes_j + 20 * global_mean_fraude) / (cnt_j + 20),
    .groups      = "drop"
  ) %>%
  select(j, cnt_j, target_enc_j)

stats_prod_i <- df_train_val %>%
  filter(!is.na(i)) %>%
  count(i, name = "cnt_i")

preparar_datos_completos <- function(data, st_a, st_j, st_i, mean_fraude, prom_monto) {
  data %>%
    mutate(
      id_fila_orig = row_number(),
      across(any_of(c("b", "c", "d", "f", "h", "l", "m", "n")), ~ na_if(., -1)),
      fecha_dt     = ymd_hms(fecha)
    ) %>%
    arrange(fecha_dt) %>%
    group_by(i) %>%
    mutate(
      segs_prev_i     = as.numeric(difftime(fecha_dt, lag(fecha_dt), units = "secs")),
      segs_prev_i     = if_else(is.na(segs_prev_i), 999999, segs_prev_i),
      log_segs_prev_i = log1p(pmax(segs_prev_i, 0)),
      rafaga_i_5min   = if_else(segs_prev_i <= 300, 1, 0)
    ) %>%
    ungroup() %>%
    group_by(j) %>%
    mutate(
      segs_prev_j     = as.numeric(difftime(fecha_dt, lag(fecha_dt), units = "secs")),
      segs_prev_j     = if_else(is.na(segs_prev_j), 999999, segs_prev_j),
      log_segs_prev_j = log1p(pmax(segs_prev_j, 0))
    ) %>%
    ungroup() %>%
    arrange(id_fila_orig) %>%
    select(-id_fila_orig) %>%
    left_join(st_a, by = "a") %>%
    left_join(st_j, by = "j") %>%
    left_join(st_i, by = "i") %>%
    mutate(
      target_enc_a      = if_else(is.na(target_enc_a), mean_fraude, target_enc_a),
      target_enc_j      = if_else(is.na(target_enc_j), mean_fraude, target_enc_j),
      cnt_j             = if_else(is.na(cnt_j), 0L, cnt_j),
      cnt_i             = if_else(is.na(cnt_i), 0L, cnt_i),
      monto_prom_cat    = if_else(is.na(monto_prom_cat), prom_monto, monto_prom_cat),
      monto_ratio_a_log = log1p(monto / monto_prom_cat),

      monto_log        = log1p(monto),
      hora             = hour(fecha_dt),
      dia_semana       = wday(fecha_dt, week_start = 1),
      es_madrugada     = if_else(hora >= 1 & hora <= 4, 1, 0),
      es_fin_de_semana = if_else(dia_semana >= 6, 1, 0),

      grado_orfandad = rowSums(is.na(across(any_of(c("b", "c", "d", "f", "g", "l", "m", "o"))))),

      o_estado = factor(if_else(is.na(o) | trimws(as.character(o)) == "", "Nulo", as.character(o))),
      p_estado = factor(if_else(is.na(p) | trimws(as.character(p)) == "", "Nulo", as.character(p))),

      g_clean = if_else(is.na(g) | trimws(as.character(g)) == "", "Otros", as.character(g)),
      g_grupo = factor(case_when(
        g_clean %in% c("BR", "AR", "UY", "US") ~ g_clean,
        TRUE ~ "Otros"
      )),

      score_decil      = ntile(score, 10),
      score_decil_fact = factor(score_decil),
      score_tramo_j    = factor(case_when(
        score_decil <= 3 ~ "Bajo_Riesgo",
        score_decil >= 8 ~ "Alto_Riesgo",
        TRUE             ~ "Zona_Intermedia"
      )),

      # Polinomios, Ratios e Interacciones
      score_cuadratico            = score^2,
      ratio_monto_score           = monto_log / (score + 1),
      ratio_monto_c               = monto / (c + 1),
      interaccion_g_madrugada     = factor(paste0(g_grupo, "_mad_", es_madrugada)),
      interaccion_score_orfandad  = score * grado_orfandad,
      interaccion_monto_madrugada = monto_log * es_madrugada,

      # Nuevos Ratios e Interacciones Puras para XGBoost Tuned L1/L2
      score_x_monto    = score * monto_log,
      ratio_l_m        = l / (m + 1),
      ratio_b_score    = b / (score + 1),
      es_monto_redondo = if_else(monto %% 10 == 0, 1, 0)
    )
}

df_tr     <- preparar_datos_completos(df_train_val, stats_cat_a, stats_cat_j, stats_prod_i, global_mean_fraude, monto_prom_global)
df_val_f  <- preparar_datos_completos(df_val, stats_cat_a, stats_cat_j, stats_prod_i, global_mean_fraude, monto_prom_global)
df_test_f <- preparar_datos_completos(test_set, stats_cat_a, stats_cat_j, stats_prod_i, global_mean_fraude, monto_prom_global)

# Alinear niveles de factores entre Train, Validation y Test
for (col_f in c("o_estado", "p_estado", "g_grupo", "score_decil_fact", "score_tramo_j", "interaccion_g_madrugada")) {
  df_val_f[[col_f]]  <- factor(df_val_f[[col_f]], levels = levels(df_tr[[col_f]]))
  df_test_f[[col_f]] <- factor(df_test_f[[col_f]], levels = levels(df_tr[[col_f]]))
}

# Dataset ROSE solo para GLM 4
set.seed(42)
df_tr_rose <- ovun.sample(
  fraude ~ monto_log + hora + es_madrugada + es_fin_de_semana + grado_orfandad +
    score + o_estado + g_grupo + score_tramo_j + interaccion_g_madrugada,
  data = df_tr, method = "both", p = 0.25, N = nrow(df_tr)
)$data

# ==============================================================================
# FAMILIA 1: REGRESIÓN LOGÍSTICA (GLM + ELASTIC NET) - VALIDATION Y TEST
# ==============================================================================
suppressWarnings({
  glm_0 <- glm(fraude ~ monto_log + hora + es_madrugada + es_fin_de_semana + grado_orfandad + score + o_estado, data = df_tr, family = binomial)
  glm_1 <- glm(fraude ~ monto_log + hora + es_madrugada + es_fin_de_semana + grado_orfandad + score + o_estado + g_grupo, data = df_tr, family = binomial)
  glm_2 <- glm(fraude ~ monto_log + hora + es_madrugada + es_fin_de_semana + grado_orfandad + score + o_estado + g_grupo + score_tramo_j, data = df_tr, family = binomial)
  glm_3 <- glm(fraude ~ monto_log + hora + es_madrugada + es_fin_de_semana + grado_orfandad + score + o_estado + g_grupo + score_tramo_j + interaccion_g_madrugada, data = df_tr, family = binomial)
  glm_4 <- glm(fraude ~ monto_log + hora + es_madrugada + es_fin_de_semana + grado_orfandad + score + o_estado + g_grupo + score_tramo_j + interaccion_g_madrugada, data = df_tr_rose, family = binomial)

  glm_5 <- glm(
    fraude ~ monto_log + hora + es_madrugada + es_fin_de_semana + grado_orfandad +
      score + score_cuadratico + o_estado + g_grupo + score_tramo_j +
      interaccion_g_madrugada + interaccion_score_orfandad +
      interaccion_monto_madrugada + monto_ratio_a_log,
    data = df_tr, family = binomial
  )

  glm_opt <- glm(
    fraude ~ monto_log + hora + es_madrugada + es_fin_de_semana + grado_orfandad +
      score + score_decil_fact + o_estado + g_grupo + target_enc_a +
      interaccion_g_madrugada + interaccion_score_orfandad + interaccion_monto_madrugada,
    data = df_tr, family = binomial
  )

  glm_vel <- glm(
    fraude ~ monto_log + hora + es_madrugada + es_fin_de_semana + grado_orfandad +
      score + score_decil_fact + o_estado + g_grupo + target_enc_a + target_enc_j +
      cnt_i + cnt_j + log_segs_prev_i + log_segs_prev_j + rafaga_i_5min +
      interaccion_g_madrugada + interaccion_score_orfandad + interaccion_monto_madrugada,
    data = df_tr, family = binomial
  )
})

formula_glmnet <- fraude ~ monto_log + hora + es_madrugada + es_fin_de_semana +
  grado_orfandad + score + score_cuadratico + score_decil_fact + o_estado +
  g_grupo + target_enc_a + monto_ratio_a_log + interaccion_g_madrugada +
  interaccion_score_orfandad + interaccion_monto_madrugada

X_tr_mat   <- model.matrix(formula_glmnet, data = df_tr)[, -1]
X_val_mat  <- model.matrix(formula_glmnet, data = df_val_f)[, -1]
X_test_mat <- model.matrix(formula_glmnet, data = df_test_f)[, -1]

set.seed(42)
cv_glmnet <- cv.glmnet(x = X_tr_mat, y = df_tr$fraude, family = "binomial", alpha = 0.5, nfolds = 5)
probs_glmnet_val <- as.numeric(predict(cv_glmnet, newx = X_val_mat, s = "lambda.min", type = "response"))
probs_glmnet_ts  <- as.numeric(predict(cv_glmnet, newx = X_test_mat, s = "lambda.min", type = "response"))

tabla_glm_val <- bind_rows(
  evaluar_modelo("GLM 0: Baseline", predict(glm_0, df_val_f, type = "response"), df_val_f$fraude),
  evaluar_modelo("GLM 1: + Geografía (g_grupo)", predict(glm_1, df_val_f, type = "response"), df_val_f$fraude),
  evaluar_modelo("GLM 2: + Tramo J Score", predict(glm_2, df_val_f, type = "response"), df_val_f$fraude),
  evaluar_modelo("GLM 3: + Interacción Riesgo", predict(glm_3, df_val_f, type = "response"), df_val_f$fraude),
  evaluar_modelo("GLM 4: + Rebalanceo ROSE", predict(glm_4, df_val_f, type = "response"), df_val_f$fraude),
  evaluar_modelo("GLM 5: + Polinomios y Ratios", predict(glm_5, df_val_f, type = "response"), df_val_f$fraude),
  evaluar_modelo("GLM Opt: Deciles + TargetEnc", predict(glm_opt, df_val_f, type = "response"), df_val_f$fraude),
  evaluar_modelo("GLM 6: + Velocity + TE(j) + Freq(i)", predict(glm_vel, df_val_f, type = "response"), df_val_f$fraude),
  evaluar_modelo("GLM 7: Regularizado (Elastic Net)", probs_glmnet_val, df_val_f$fraude)
) %>% arrange(desc(AUC_PR))

tabla_glm_test <- bind_rows(
  evaluar_en_test("GLM 0: Baseline", predict(glm_0, df_test_f, type = "response"), df_test_f$fraude, obtener_umbral(tabla_glm_val, "GLM 0: Baseline")),
  evaluar_en_test("GLM 1: + Geografía (g_grupo)", predict(glm_1, df_test_f, type = "response"), df_test_f$fraude, obtener_umbral(tabla_glm_val, "GLM 1: + Geografía (g_grupo)")),
  evaluar_en_test("GLM 2: + Tramo J Score", predict(glm_2, df_test_f, type = "response"), df_test_f$fraude, obtener_umbral(tabla_glm_val, "GLM 2: + Tramo J Score")),
  evaluar_en_test("GLM 3: + Interacción Riesgo", predict(glm_3, df_test_f, type = "response"), df_test_f$fraude, obtener_umbral(tabla_glm_val, "GLM 3: + Interacción Riesgo")),
  evaluar_en_test("GLM 4: + Rebalanceo ROSE", predict(glm_4, df_test_f, type = "response"), df_test_f$fraude, obtener_umbral(tabla_glm_val, "GLM 4: + Rebalanceo ROSE")),
  evaluar_en_test("GLM 5: + Polinomios y Ratios", predict(glm_5, df_test_f, type = "response"), df_test_f$fraude, obtener_umbral(tabla_glm_val, "GLM 5: + Polinomios y Ratios")),
  evaluar_en_test("GLM Opt: Deciles + TargetEnc", predict(glm_opt, df_test_f, type = "response"), df_test_f$fraude, obtener_umbral(tabla_glm_val, "GLM Opt: Deciles + TargetEnc")),
  evaluar_en_test("GLM 6: + Velocity + TE(j) + Freq(i)", predict(glm_vel, df_test_f, type = "response"), df_test_f$fraude, obtener_umbral(tabla_glm_val, "GLM 6: + Velocity + TE(j) + Freq(i)")),
  evaluar_en_test("GLM 7: Regularizado (Elastic Net)", probs_glmnet_ts, df_test_f$fraude, obtener_umbral(tabla_glm_val, "GLM 7: Regularizado (Elastic Net)"))
) %>% arrange(desc(AUC_PR_Test))

print("=== 1A. VALIDATION - FAMILIA REGRESIÓN LOGÍSTICA ===")
print(tabla_glm_val)
print("=== 1B. TEST - FAMILIA REGRESIÓN LOGÍSTICA ===")
print(tabla_glm_test)

# ==============================================================================
# FAMILIA 2: ÁRBOLES DE DECISIÓN INDIVIDUALES (CART) - VALIDATION Y TEST
# ==============================================================================
formula_arbol_full <- fraude ~ monto_log + hora + es_madrugada + es_fin_de_semana +
  grado_orfandad + score + o_estado + p_estado + g_grupo + target_enc_a +
  b + f + l + m + n + ratio_monto_score + ratio_monto_c +
  interaccion_score_orfandad + interaccion_monto_madrugada

arbol_prepoda_6 <- rpart(
  formula = formula_arbol_full,
  data    = df_tr,
  method  = "class",
  parms   = list(split = "gini"),
  control = rpart.control(maxdepth = 6, minbucket = 20, minsplit = 40, cp = 0.0005)
)

arbol_alta_res_v2 <- rpart(
  formula = formula_arbol_full,
  data    = df_tr,
  method  = "class",
  parms   = list(split = "gini"),
  control = rpart.control(maxdepth = 10, minbucket = 40, minsplit = 80, cp = 0.00002, maxsurrogate = 5)
)

arbol_laplace_v2 <- arbol_alta_res_v2
n_total_hoja     <- arbol_laplace_v2$frame$n
n_pos_hoja       <- arbol_laplace_v2$frame$yval2[, 3]
arbol_laplace_v2$frame$yval2[, 5] <- (n_pos_hoja + 1) / (n_total_hoja + 2)

tabla_arboles_val <- bind_rows(
  evaluar_modelo("Árbol 1: Pre-poda (maxdepth=6, minbucket=20)", predict(arbol_prepoda_6, df_val_f, type = "prob")[, "1"], df_val_f$fraude),
  evaluar_modelo("Árbol 2: Alta Res + Vars Completas (cp=2e-5)", predict(arbol_alta_res_v2, df_val_f, type = "prob")[, "1"], df_val_f$fraude),
  evaluar_modelo("Árbol 3: Alta Res + Vars Completas + Laplace", predict(arbol_laplace_v2, df_val_f, type = "prob")[, "1"], df_val_f$fraude)
) %>% arrange(desc(AUC_PR))

tabla_arboles_test <- bind_rows(
  evaluar_en_test("Árbol 1: Pre-poda (maxdepth=6, minbucket=20)", predict(arbol_prepoda_6, df_test_f, type = "prob")[, "1"], df_test_f$fraude, obtener_umbral(tabla_arboles_val, "Árbol 1: Pre-poda (maxdepth=6, minbucket=20)")),
  evaluar_en_test("Árbol 2: Alta Res + Vars Completas (cp=2e-5)", predict(arbol_alta_res_v2, df_test_f, type = "prob")[, "1"], df_test_f$fraude, obtener_umbral(tabla_arboles_val, "Árbol 2: Alta Res + Vars Completas (cp=2e-5)")),
  evaluar_en_test("Árbol 3: Alta Res + Vars Completas + Laplace", predict(arbol_laplace_v2, df_test_f, type = "prob")[, "1"], df_test_f$fraude, obtener_umbral(tabla_arboles_val, "Árbol 3: Alta Res + Vars Completas + Laplace"))
) %>% arrange(desc(AUC_PR_Test))

print("=== 2A. VALIDATION - FAMILIA ÁRBOLES DE DECISIÓN (CART) ===")
print(tabla_arboles_val)
print("=== 2B. TEST - FAMILIA ÁRBOLES DE DECISIÓN (CART) ===")
print(tabla_arboles_test)

# ==============================================================================
# FAMILIA 3: ENSAMBLES DE ÁRBOLES (RF + XGBOOST BASE + XGBOOST TUNED L1/L2)
# ==============================================================================

# --- 3.1 Random Forest (300 árboles) ---
vars_rf <- c("monto_log", "hora", "es_madrugada", "es_fin_de_semana", "grado_orfandad",
             "score", "o_estado", "p_estado", "g_grupo", "target_enc_a",
             "b", "f", "l", "m", "n", "ratio_monto_score", "ratio_monto_c",
             "interaccion_score_orfandad", "interaccion_monto_madrugada")

df_tr_rf   <- df_tr[, c("fraude", vars_rf)]
df_val_rf  <- df_val_f[, c("fraude", vars_rf)]
df_test_rf <- df_test_f[, c("fraude", vars_rf)]

for (col in vars_rf) {
  if (is.numeric(df_tr_rf[[col]])) {
    med_val <- median(df_tr_rf[[col]], na.rm = TRUE)
    df_tr_rf[[col]][is.na(df_tr_rf[[col]])]     <- med_val
    df_val_rf[[col]][is.na(df_val_rf[[col]])]   <- med_val
    df_test_rf[[col]][is.na(df_test_rf[[col]])] <- med_val
  }
}
df_tr_rf$fraude <- factor(df_tr_rf$fraude, levels = c(0, 1))

set.seed(42)
modelo_rf <- ranger(
  formula = fraude ~ ., data = df_tr_rf, num.trees = 300, mtry = 4,
  min.node.size = 25, probability = TRUE, num.threads = parallel::detectCores() - 1
)
probs_rf_val <- predict(modelo_rf, data = df_val_rf)$predictions[, "1"]
probs_rf_ts  <- predict(modelo_rf, data = df_test_rf)$predictions[, "1"]

# --- 3.2 XGBoost Base ---
vars_xgb <- c("monto_log", "hora", "es_madrugada", "es_fin_de_semana", "grado_orfandad",
              "score", "o_estado", "p_estado", "g_grupo", "target_enc_a",
              "b", "c", "d", "f", "h", "l", "m", "n",
              "ratio_monto_score", "ratio_monto_c",
              "interaccion_score_orfandad", "interaccion_monto_madrugada")

options(na.action = "na.pass")
X_tr_xgb   <- model.matrix(~ . - 1, data = df_tr[, vars_xgb])
X_val_xgb  <- model.matrix(~ . - 1, data = df_val_f[, vars_xgb])
X_test_xgb <- model.matrix(~ . - 1, data = df_test_f[, vars_xgb])
options(na.action = "na.omit")

dtrain <- xgb.DMatrix(data = X_tr_xgb, label = df_tr$fraude)
dval   <- xgb.DMatrix(data = X_val_xgb, label = df_val_f$fraude)
dtest  <- xgb.DMatrix(data = X_test_xgb, label = df_test_f$fraude)

params_xgb <- list(
  objective = "binary:logistic", eval_metric = "aucpr", max_depth = 5,
  eta = 0.05, subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 20
)

set.seed(42)
modelo_xgb <- xgb.train(
  params = params_xgb, data = dtrain, nrounds = 400,
  watchlist = list(val = dval), early_stopping_rounds = 30, verbose = 0
)
probs_xgb_val <- predict(modelo_xgb, newdata = dval)
probs_xgb_ts  <- predict(modelo_xgb, newdata = dtest)

# --- 3.3 XGBoost Tuned L1/L2 + Nuevos Ratios ---
vars_xgb_v2 <- c(vars_xgb, "monto_ratio_a_log", "ratio_l_m", "ratio_b_score",
                 "score_x_monto", "es_monto_redondo")

options(na.action = "na.pass")
X_tr_2   <- model.matrix(~ . - 1, data = df_tr[, vars_xgb_v2])
X_val_2  <- model.matrix(~ . - 1, data = df_val_f[, vars_xgb_v2])
X_test_2 <- model.matrix(~ . - 1, data = df_test_f[, vars_xgb_v2])
options(na.action = "na.omit")

dtrain_2 <- xgb.DMatrix(data = X_tr_2, label = df_tr$fraude)
dval_2   <- xgb.DMatrix(data = X_val_2, label = df_val_f$fraude)
dtest_2  <- xgb.DMatrix(data = X_test_2, label = df_test_f$fraude)

params_xgb_tuned <- list(
  objective         = "binary:logistic",
  eval_metric       = "aucpr",
  max_depth         = 6,
  eta               = 0.025,
  subsample         = 0.80,
  colsample_bytree  = 0.75,
  colsample_bylevel = 0.75,
  min_child_weight  = 25,
  lambda            = 2.0,
  alpha             = 0.5
)

set.seed(42)
xgb_tuned <- xgb.train(
  params                = params_xgb_tuned,
  data                  = dtrain_2,
  nrounds               = 800,
  watchlist             = list(val = dval_2),
  early_stopping_rounds = 45,
  verbose               = 0
)
probs_xgb2_val <- predict(xgb_tuned, newdata = dval_2)
probs_xgb2_ts  <- predict(xgb_tuned, newdata = dtest_2)

tabla_ensambles_val <- bind_rows(
  evaluar_modelo("Random Forest (300 árboles)", probs_rf_val, df_val_f$fraude),
  evaluar_modelo("XGBoost Base (Gradient Boosting)", probs_xgb_val, df_val_f$fraude),
  evaluar_modelo("XGBoost Tuned L1/L2 + Nuevos Ratios", probs_xgb2_val, df_val_f$fraude)
) %>% arrange(desc(AUC_PR))

tabla_ensambles_test <- bind_rows(
  evaluar_en_test("Random Forest (300 árboles)", probs_rf_ts, df_test_f$fraude, obtener_umbral(tabla_ensambles_val, "Random Forest (300 árboles)")),
  evaluar_en_test("XGBoost Base (Gradient Boosting)", probs_xgb_ts, df_test_f$fraude, obtener_umbral(tabla_ensambles_val, "XGBoost Base (Gradient Boosting)")),
  evaluar_en_test("XGBoost Tuned L1/L2 + Nuevos Ratios", probs_xgb2_ts, df_test_f$fraude, obtener_umbral(tabla_ensambles_val, "XGBoost Tuned L1/L2 + Nuevos Ratios"))
) %>% arrange(desc(AUC_PR_Test))

print("=== 3A. VALIDATION - FAMILIA ENSAMBLES (UMBRAL LIBRE) ===")
print(tabla_ensambles_val)
print("=== 3B. TEST - FAMILIA ENSAMBLES (UMBRAL LIBRE) ===")
print(tabla_ensambles_test)


# ==============================================================================
# MEJORA DE RANDOM FOREST (INDICADORES NA + NUEVOS RATIOS + HELLINGER + UMBRAL FINO)
# ==============================================================================

# 1. Agregar flags de valores faltantes antes de imputar (conserva la señal de NA)
agregar_flags_na <- function(df) {
  df %>%
    mutate(
      es_na_b = if_else(is.na(b), 1L, 0L),
      es_na_c = if_else(is.na(c), 1L, 0L),
      es_na_d = if_else(is.na(d), 1L, 0L),
      es_na_l = if_else(is.na(l), 1L, 0L),
      es_na_m = if_else(is.na(m), 1L, 0L)
    )
}

df_tr_rf_m   <- agregar_flags_na(df_tr)
df_val_rf_m  <- agregar_flags_na(df_val_f)
df_test_rf_m <- agregar_flags_na(df_test_f)

# 2. Vector ampliado (las 27 vars de XGBoost Tuned + 5 flags de nulidad = 32 vars)
vars_rf_opt <- c(
  "monto_log", "hora", "es_madrugada", "es_fin_de_semana", "grado_orfandad",
  "score", "o_estado", "p_estado", "g_grupo", "target_enc_a", "monto_ratio_a_log",
  "b", "c", "d", "f", "h", "l", "m", "n",
  "ratio_monto_score", "ratio_monto_c", "ratio_l_m", "ratio_b_score",
  "score_x_monto", "es_monto_redondo",
  "interaccion_score_orfandad", "interaccion_monto_madrugada",
  "es_na_b", "es_na_c", "es_na_d", "es_na_l", "es_na_m"
)

df_tr_rf_m   <- df_tr_rf_m[, c("fraude", vars_rf_opt)]
df_val_rf_m  <- df_val_rf_m[, c("fraude", vars_rf_opt)]
df_test_rf_m <- df_test_rf_m[, c("fraude", vars_rf_opt)]

# Imputar numéricas con la mediana de Train (ahora sin perder la información del NA)
for (col in vars_rf_opt) {
  if (is.numeric(df_tr_rf_m[[col]])) {
    med_val <- median(df_tr_rf_m[[col]], na.rm = TRUE)
    df_tr_rf_m[[col]][is.na(df_tr_rf_m[[col]])]     <- med_val
    df_val_rf_m[[col]][is.na(df_val_rf_m[[col]])]   <- med_val
    df_test_rf_m[[col]][is.na(df_test_rf_m[[col]])] <- med_val
  }
}
df_tr_rf_m$fraude <- factor(df_tr_rf_m$fraude, levels = c(0, 1))

# 3A. Random Forest Tuned (Gini + 500 árboles + Flags NA + 32 vars)
set.seed(42)
rf_tuned_gini <- ranger(
  formula       = fraude ~ .,
  data          = df_tr_rf_m,
  num.trees     = 500,
  mtry          = 6,
  min.node.size = 15,
  splitrule     = "gini",
  probability   = TRUE,
  num.threads   = parallel::detectCores() - 1
)
probs_rf_gini_val  <- predict(rf_tuned_gini, data = df_val_rf_m)$predictions[, "1"]
probs_rf_gini_test <- predict(rf_tuned_gini, data = df_test_rf_m)$predictions[, "1"]

# 3B. Random Forest Desbalanceado (Splitrule Hellinger + 500 árboles + Flags NA)
set.seed(42)
rf_tuned_hellinger <- ranger(
  formula       = fraude ~ .,
  data          = df_tr_rf_m,
  num.trees     = 500,
  mtry          = 6,
  min.node.size = 15,
  splitrule     = "hellinger",
  probability   = TRUE,
  num.threads   = parallel::detectCores() - 1
)
probs_rf_hell_val  <- predict(rf_tuned_hellinger, data = df_val_rf_m)$predictions[, "1"]
probs_rf_hell_test <- predict(rf_tuned_hellinger, data = df_test_rf_m)$predictions[, "1"]

# 4. Función con grilla fina de umbral [0.005, 0.150] para minimizar el Costo
evaluar_rf_fino <- function(nombre, probs, y_real, c_fp = 5, c_fn = 200) {
  y_num <- as.numeric(as.character(y_real))
  pr <- pr.curve(scores.class0 = probs[y_num == 1], scores.class1 = probs[y_num == 0], curve = FALSE)

  umbrales <- unique(c(seq(0.005, 0.150, by = 0.001), seq(0.16, 0.95, by = 0.01)))
  costos <- sapply(umbrales, function(u) {
    preds <- ifelse(probs >= u, 1, 0)
    (sum(preds == 1 & y_num == 0) * c_fp) + (sum(preds == 0 & y_num == 1) * c_fn)
  })

  u_opt <- umbrales[which.min(costos)]
  preds_opt <- ifelse(probs >= u_opt, 1, 0)
  fp <- sum(preds_opt == 1 & y_num == 0)
  fn <- sum(preds_opt == 0 & y_num == 1)
  tp <- sum(preds_opt == 1 & y_num == 1)

  data.frame(
    Modelo        = nombre,
    AUC_PR        = round(pr$auc.integral, 4),
    Umbral_Optimo = u_opt,
    Precision     = round(tp / (tp + fp), 4),
    Recall        = round(tp / (tp + fn), 4),
    FP            = fp,
    FN            = fn,
    Costo_Total   = min(costos)
  )
}

# 5. Comparativa en Validation y Test (RF Base vs RF Optimizado)
tabla_rf_val <- bind_rows(
  evaluar_modelo("RF 1: Base (300 árboles, 19 vars)", probs_rf_val, df_val_f$fraude),
  evaluar_rf_fino("RF 2: Tuned Gini + Flags NA + 32 vars", probs_rf_gini_val, df_val_f$fraude),
  evaluar_rf_fino("RF 3: Tuned Hellinger + Flags NA + 32 vars", probs_rf_hell_val, df_val_f$fraude)
) %>% arrange(desc(AUC_PR))

tabla_rf_test <- bind_rows(
  evaluar_en_test("RF 1: Base (300 árboles, 19 vars)", probs_rf_ts, df_test_f$fraude, obtener_umbral(tabla_rf_val, "RF 1: Base (300 árboles, 19 vars)")),
  evaluar_en_test("RF 2: Tuned Gini + Flags NA + 32 vars", probs_rf_gini_test, df_test_f$fraude, obtener_umbral(tabla_rf_val, "RF 2: Tuned Gini + Flags NA + 32 vars")),
  evaluar_en_test("RF 3: Tuned Hellinger + Flags NA + 32 vars", probs_rf_hell_test, df_test_f$fraude, obtener_umbral(tabla_rf_val, "RF 3: Tuned Hellinger + Flags NA + 32 vars"))
) %>% arrange(desc(AUC_PR_Test))

print("=== COMPARATIVA RANDOM FOREST EN VALIDATION ===")
print(tabla_rf_val)
print("=== COMPARATIVA RANDOM FOREST EN TEST ===")
print(tabla_rf_test)
# ==============================================================================
# BLOQUE D: GRÁFICOS LEGIBLES PARA LA PRESENTACIÓN
# ==============================================================================
arbol_visual_3 <- rpart(
  formula = formula_arbol_full,
  data    = df_tr,
  method  = "class",
  control = rpart.control(maxdepth = 3, minbucket = 100, cp = 0.0005)
)

rpart.plot(
  arbol_visual_3,
  type          = 4,
  extra         = 104,
  fallen.leaves = TRUE,
  cex           = 0.85,
  tweak         = 1.1,
  box.palette   = "RdBu",
  shadow.col    = "gray",
  main          = "Reglas Principales de Decisión (Primeros 3 Niveles)"
)

imp_df <- data.frame(
  Variable    = names(arbol_alta_res_v2$variable.importance),
  Importancia = as.numeric(arbol_alta_res_v2$variable.importance)
) %>%
  mutate(Importancia_Relativa = (Importancia / sum(Importancia)) * 100) %>%
  arrange(desc(Importancia_Relativa)) %>%
  slice(1:12)

ggplot(imp_df, aes(x = reorder(Variable, Importancia_Relativa), y = Importancia_Relativa)) +
  geom_col(fill = "#2C3E50", width = 0.7) +
  geom_text(aes(label = paste0(round(Importancia_Relativa, 1), "%")), hjust = -0.15, size = 3.8) +
  coord_flip() +
  ylim(0, max(imp_df$Importancia_Relativa) * 1.15) +
  labs(
    title    = "Top 12 Variables Más Importantes en el Árbol CART",
    subtitle = "Contribución relativa a la reducción de impureza de Gini",
    x        = "",
    y        = "Importancia Relativa (%)"
  ) +
  theme_minimal(base_size = 13)
