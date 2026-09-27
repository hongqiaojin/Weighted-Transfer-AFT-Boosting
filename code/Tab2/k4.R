# 4 source domains
library(xgboost)
library(survival)
library(doParallel)
library(foreach)
library(dplyr)
library(tidyr)
library(ggplot2)

# --- 0. Parameters ---
set.seed(123)
TARGET_TOTAL_SIZES <- c(100, 200, 300) 
SOURCE_SIZES <- c(100, 300, 500, 1000)       
NUM_REPLICATIONS <- 500 
TRAIN_TEST_RATIO <- 0.6  

# --- 1. Objective and Eval Functions ---
myobject <- function(preds, dtrain){
  T <- getinfo(dtrain,"label"); d <- attr(dtrain,'censor')
  u <- (log(T) - preds); fai <- dnorm(u); Fai <- pnorm(u)
  g <- outer(fai, (1 - Fai), '/'); g <- as.matrix(g)
  grad <- -(d * u) - g %*% (1 - d)
  hess <- d + ((1 - d) %*% g)
  return(list(grad = grad, hess = hess))
}

evalerror <- function(preds, dtrain){
  T <- getinfo(dtrain,"label"); d <- attr(dtrain,"censor")
  u <- (log(T) - preds); fai <- dnorm(u); Fai <- pnorm(u)
  n <- d * log((1/(T)) * fai); m <- (1 - d) * log(1 - Fai)
  err <- -as.numeric(sum(m + n))
  return(list(metric = "deviance", value = err))
}

generate_data <- function(n, shift_vec) {
  x <- matrix(rnorm(n * 6), ncol = 6)
  v1 <- rnorm(n, 0, 1)
  real.time <- exp(
    sin(x[,1] + 1 + shift_vec[1]) + 
      (1 + shift_vec[2]) * x[,2] * x[,3] - 
      (1 + shift_vec[3]) * x[,4]**2 - 
      (0.3 + shift_vec[5]) * abs(x[,5]) - 
      (0.2 + shift_vec[6]) * x[,6] + 
      shift_vec[4] * x[,4] + v1
  )
  cens.time <- runif(n, min = 0, max = 2)
  status <- ifelse(real.time <= cens.time, 1, 0)
  obs.time <- ifelse(real.time <= cens.time, real.time, cens.time)
  return(list(data = x, label = obs.time, censor = status))
}

param <- list(max_depth = 3, eta = 0.035, objective = myobject, eval_metric = evalerror, alpha = 1)
nround <- 500 

# --- 2. Expanded Experiment Logic (4 Source Domains) ---
run_compare_experiment <- function(target_total_size, source_size, param, nround, ratio) {
  target_train_size <- round(target_total_size * ratio)
  D_target <- generate_data(target_total_size, rep(0, 6))
  
  idx_train <- 1:target_train_size
  idx_test <- (target_train_size + 1):target_total_size
  
  D_target_train_xgb <- xgb.DMatrix(D_target$data[idx_train,], label = D_target$label[idx_train])
  attr(D_target_train_xgb, "censor") <- D_target$censor[idx_train]
  D_target_test_xgb <- xgb.DMatrix(D_target$data[idx_test,], label = D_target$label[idx_test])
  attr(D_target_test_xgb, "censor") <- D_target$censor[idx_test]
  test_surv <- Surv(D_target$label[idx_test], D_target$censor[idx_test])
  
  # Define 4 different Source Shifts
  shifts <- list(
    s1 = c(-1, -0.8, -0.5, 0.5, -0.3, 0),
    s2 = c(1, 0.8, 0.6, -0.6, -0.4, -0.2),
    s3 = c(0.5, -0.5, 1, -1, 0.2, -0.2),
    s4 = c(-0.5, 0.5, -1, 1, -0.2, 0.2)
  )
  
  models <- list()
  models[["Target"]] <- xgb.train(param, D_target_train_xgb, nround = nround)
  
  for(i in 1:4) {
    d_raw <- generate_data(source_size, shifts[[i]])
    d_xgb <- xgb.DMatrix(d_raw$data, label = d_raw$label)
    attr(d_xgb, "censor") <- d_raw$censor
    models[[paste0("S", i)]] <- xgb.train(param, d_xgb, nround = nround)
  }
  
  # Calculate Weights (Concordance on Target Train set)
  weights <- sapply(models, function(m) {
    p <- predict(m, D_target_train_xgb)
    concordance(Surv(D_target$label[idx_train], D_target$censor[idx_train]) ~ exp(p))$concordance[1]
  })
  w_norm <- weights / sum(weights)
  
  # Individual Model C-index on Test Set
  c_indices <- sapply(models, function(m) {
    concordance(test_surv ~ exp(predict(m, D_target_test_xgb)))$concordance[1]
  })
  
  # Weighted Transfer Prediction
  p_matrix <- sapply(models, function(m) predict(m, D_target_test_xgb))
  p_weighted <- p_matrix %*% w_norm
  c_weighted <- concordance(test_surv ~ exp(p_weighted))$concordance[1]
  
  results <- c(c_indices, Weighted = c_weighted)
  return(results)
}

# --- 3. Parallel Processing ---
num_cores <- 3
cl <- makeCluster(num_cores)
registerDoParallel(cl)

exp_grid <- expand.grid(target_size = TARGET_TOTAL_SIZES, source_size = SOURCE_SIZES, rep = 1:NUM_REPLICATIONS)

results_raw <- foreach(i = 1:nrow(exp_grid), .packages = c("xgboost", "survival"), .combine = "rbind") %dopar% {
  res <- run_compare_experiment(exp_grid$target_size[i], exp_grid$source_size[i], param, nround, TRAIN_TEST_RATIO)
  data.frame(Target_Size = exp_grid$target_size[i], Source_Size = exp_grid$source_size[i], 
             Method = names(res), CIndex = as.numeric(res))
}
stopCluster(cl)

# --- 4. Table Generation (Mean and SD formatted) ---
library(tidyverse)

library(tidyverse)
write.csv(results_raw, file = "C:/Users/lenovo/Desktop/AFTBoost/k4/results_raw.csv", row.names = FALSE)# Convert multiple values into a vector
library(tidyverse)
summary_table <- results_raw %>%
  group_by(Method, Target_Size, Source_Size) %>%
  summarise(
    Mean = as.character(round(mean(CIndex), 4)),
    SD = paste0("(", round(sd(CIndex), 4), ")"),
    .groups = "drop"
  ) %>%
  # 1. Create sample size combination labels
  mutate(Condition = paste0("T:", Target_Size, "/S:", Source_Size)) %>%
  # 2. Convert Mean and SD to long format so they appear as separate rows
  pivot_longer(cols = c(Mean, SD), names_to = "Stat Type", values_to = "Value") %>%
  # 3. Reshape the table: columns are Methods, rows are Condition and statistic type
  pivot_wider(names_from = Method, values_from = Value) %>%
  # 4. Optional: keep repeated Condition values arranged for better readability after export
  select(Condition, `Stat Type`, everything())

# Print results for inspection
print(summary_table)
data <- as.data.frame(summary_table)

# Print results
print(data)
write.csv(data, file = "C:/Users/lenovo/Desktop/AFTBoost/k4/k4.csv", row.names = FALSE)# Convert multiple values into a vector
# Save if needed
# write.csv(data, "model_performance_transposed.csv", row.names = FALSE)