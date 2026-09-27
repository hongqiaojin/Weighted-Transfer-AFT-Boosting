library(xgboost)
library(survival)
library(doParallel)
library(foreach)
library(dplyr)
library(ggplot2)
library(grid)

# --- 0. Experimental parameters ---
set.seed(123)

TARGET_TOTAL_SIZES <- c(100, 200, 300)
SOURCE_SIZES <- c(100, 300, 500, 1000)

NUM_REPLICATIONS <- 500
TRAIN_TEST_RATIO <- 0.6

# --- 1. Data generation and evaluation functions ---

myobject <- function(preds, dtrain) {
  T <- getinfo(dtrain, "label")
  d <- attr(dtrain, "censor")
  
  u <- log(T) - preds
  fai <- dnorm(u)
  Fai <- pnorm(u)
  
  g <- outer(fai, 1 - Fai, "/")
  g <- as.matrix(g)
  
  grad <- -(d * u) - g %*% (1 - d)
  hess <- d + (1 - d) %*% g
  
  list(
    grad = as.numeric(grad),
    hess = as.numeric(hess)
  )
}

evalerror <- function(preds, dtrain) {
  T <- getinfo(dtrain, "label")
  d <- attr(dtrain, "censor")
  
  u <- log(T) - preds
  fai <- dnorm(u)
  Fai <- pnorm(u)
  
  n <- d * log((1 / T) * fai)
  m <- (1 - d) * log(1 - Fai)
  
  err <- -sum(m + n)
  
  list(
    metric = "deviance",
    value = as.numeric(err)
  )
}

generate_data <- function(n, shift_vec) {
  x <- matrix(rnorm(n * 6), ncol = 6)
  v1 <- rnorm(n)
  
  real.time <- exp(
    sin(x[, 1] + 1 + shift_vec[1]) +
      (1 + shift_vec[2]) * x[, 2] * x[, 3] -
      (1 + shift_vec[3]) * x[, 4]^2 -
      (0.3 + shift_vec[5]) * abs(x[, 5]) -
      (0.2 + shift_vec[6]) * x[, 6] +
      shift_vec[4] * x[, 4] +
      v1
  )
  
  cens.time <- runif(n, min = 0, max = 4)
  status <- ifelse(real.time <= cens.time, 1, 0)
  obs.time <- pmin(real.time, cens.time)
  
  list(
    data = x,
    label = obs.time,
    censor = status
  )
}

param <- list(
  max_depth = 3,
  eta = 0.035,
  objective = myobject,
  eval_metric = evalerror,
  alpha = 1
)

nround <- 500

# --- 2. Single simulation function ---

run_compare_experiment <- function(
    target_total_size,
    source_size,
    param,
    nround,
    ratio
) {
  target_train_size <- round(target_total_size * ratio)
  
  D_target <- generate_data(
    target_total_size,
    rep(0, 6)
  )
  
  idx_train <- 1:target_train_size
  idx_test <- (target_train_size + 1):target_total_size
  
  D_target_train_xgb <- xgb.DMatrix(
    D_target$data[idx_train, ],
    label = D_target$label[idx_train]
  )
  attr(D_target_train_xgb, "censor") <-
    D_target$censor[idx_train]
  
  D_target_test_xgb <- xgb.DMatrix(
    D_target$data[idx_test, ],
    label = D_target$label[idx_test]
  )
  attr(D_target_test_xgb, "censor") <-
    D_target$censor[idx_test]
  
  test_surv <- Surv(
    D_target$label[idx_test],
    D_target$censor[idx_test]
  )
  
  shift_s1 <- c(-1, -0.8, -0.5, 0.5, -0.3, 0)
  shift_s2 <- c(1, 0.8, 0.6, -0.6, -0.4, -0.2)
  
  D_s1_raw <- generate_data(source_size, shift_s1)
  D_s1_xgb <- xgb.DMatrix(
    D_s1_raw$data,
    label = D_s1_raw$label
  )
  attr(D_s1_xgb, "censor") <- D_s1_raw$censor
  
  D_s2_raw <- generate_data(source_size, shift_s2)
  D_s2_xgb <- xgb.DMatrix(
    D_s2_raw$data,
    label = D_s2_raw$label
  )
  attr(D_s2_xgb, "censor") <- D_s2_raw$censor
  
  md_target <- xgb.train(param, D_target_train_xgb, nround)
  md_s1 <- xgb.train(param, D_s1_xgb, nround)
  md_s2 <- xgb.train(param, D_s2_xgb, nround)
  
  all_mds <- list(md_target, md_s1, md_s2)
  
  weights <- sapply(all_mds, function(model) {
    pred <- predict(model, D_target_train_xgb)
    
    concordance(
      Surv(
        D_target$label[idx_train],
        D_target$censor[idx_train]
      ) ~ exp(pred)
    )$concordance[1]
  })
  
  w_norm <- weights / sum(weights)
  
  pred_target <- predict(md_target, D_target_test_xgb)
  pred_s1 <- predict(md_s1, D_target_test_xgb)
  pred_s2 <- predict(md_s2, D_target_test_xgb)
  
  c_target <- concordance(
    test_surv ~ exp(pred_target)
  )$concordance[1]
  
  c_s1 <- concordance(
    test_surv ~ exp(pred_s1)
  )$concordance[1]
  
  c_s2 <- concordance(
    test_surv ~ exp(pred_s2)
  )$concordance[1]
  
  pred_weighted <-
    w_norm[1] * pred_target +
    w_norm[2] * pred_s1 +
    w_norm[3] * pred_s2
  
  c_weighted <- concordance(
    test_surv ~ exp(pred_weighted)
  )$concordance[1]
  
  c(
    Target_Only = c_target,
    Source1_Only = c_s1,
    Source2_Only = c_s2,
    Weighted = c_weighted
  )
}

# --- 3. Parallel simulation ---

num_cores <-30
cl <- makeCluster(num_cores)
registerDoParallel(cl)

exp_grid <- expand.grid(
  Target_Size = TARGET_TOTAL_SIZES,
  Source_Size = SOURCE_SIZES,
  Replication = 1:NUM_REPLICATIONS
)

results_raw <- foreach(
  i = 1:nrow(exp_grid),
  .packages = c("xgboost", "survival"),
  .combine = "rbind"
) %dopar% {
  
  res <- run_compare_experiment(
    target_total_size = exp_grid$Target_Size[i],
    source_size = exp_grid$Source_Size[i],
    param = param,
    nround = nround,
    ratio = TRAIN_TEST_RATIO
  )
  
  data.frame(
    Target_Size = exp_grid$Target_Size[i],
    Source_Size = exp_grid$Source_Size[i],
    Replication = exp_grid$Replication[i],
    Method = names(res),
    CIndex = as.numeric(res)
  )
}

stopCluster(cl)

# Save raw results
write.csv(
  results_raw,
  "C:/Users/lenovo/Desktop/TR_AFTBoost/c30/results_raw.csv",
  row.names = FALSE
)

# --- 4. Calculate mean and empirical 95% quantile interval ---

summary_plot_data <- results_raw %>%
  group_by(Target_Size, Source_Size, Method) %>%
  summarise(
    Mean_C = mean(CIndex, na.rm = TRUE),
    Lower_95 = quantile(
      CIndex,
      0.025,
      na.rm = TRUE
    ),
    Upper_95 = quantile(
      CIndex,
      0.975,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

summary_plot_data$Method <- factor(
  summary_plot_data$Method,
  levels = c(
    "Target_Only",
    "Source1_Only",
    "Source2_Only",
    "Weighted"
  ),
  labels = c(
    "Target Only",
    "S1 Only",
    "S2 Only",
    "Weighted Transfer"
  )
)

write.csv(
  summary_plot_data,
  "C:/Users/lenovo/Desktop/TR_AFTBoost/c30/summary_plot_data.csv",
  row.names = FALSE
)

# --- 5. Plot empirical 95% quantile intervals ---

p_ci <- ggplot(
  summary_plot_data,
  aes(
    x = factor(Target_Size),
    y = Mean_C,
    color = Method,
    group = 1
  )
) +
  geom_line(
    linewidth = 0.7,
    linetype = "dashed"
  ) +
  geom_errorbar(
    aes(
      ymin = Lower_95,
      ymax = Upper_95
    ),
    width = 0.15,
    linewidth = 0.7
  ) +
  geom_point(size = 2.5) +
  facet_grid(
    Source_Size ~ Method,
    labeller = labeller(
      Source_Size = function(x) {
        paste("Source N =", x)
      },
      Method = label_value
    )
  ) +
  coord_cartesian(ylim = c(0.5, 0.9)) +
  scale_color_manual(
    values = c(
      "Target Only" = "#76c7b7",
      "S1 Only" = "#f59e7d",
      "S2 Only" = "#a3b7d9",
      "Weighted Transfer" = "#e6a2c3"
    )
  ) +
  labs(
    x = "Target N",
    y = "C-index",
    title = ""
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "gray95"),
    strip.text = element_text(face = "bold", size = 9),
    panel.spacing = unit(0.5, "lines"),
    panel.grid.minor = element_blank()
  )

print(p_ci)

ggsave(
  "C:/Users/lenovo/Desktop/TR_AFTBoost/c30/c30_empirical_ci.png",
  plot = p_ci,
  width = 8,
  height = 8,
  dpi = 600
)