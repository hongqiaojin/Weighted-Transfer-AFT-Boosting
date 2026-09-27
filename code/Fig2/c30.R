library(xgboost)
library(survival)
library(doParallel)
library(foreach)
library(dplyr)
library(ggplot2)
library(tidyr)

# --- 0. Set experimental parameters ---
set.seed(123)

TARGET_TOTAL_SIZES <- c(100, 200, 300) 
SOURCE_SIZES <- c(100, 300, 500,1000)       
K_FIXED <- 2                            
NUM_REPLICATIONS <- 500 # At least 100 replications are recommended for stable error bars
TRAIN_TEST_RATIO <- 0.6  

# --- 1. Data generation and evaluation functions ---
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
      shift_vec[4] * x[,4] + 
      v1
  )
  cens.time <- runif(n, min = 0, max = 4)
  status <- ifelse(real.time <= cens.time, 1, 0)
  length(which(status==0))
  obs.time <- ifelse(real.time <= cens.time, real.time, cens.time)
  return(list(data = x, label = obs.time, censor = status))
}

param <- list(max_depth = 3, eta = 0.035, objective = myobject, eval_metric = evalerror, alpha = 1)
nround <- 500 

# --- 2. Core experimental procedure ---
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
  
  # Multivariable shift settings
  shift_s1 <- c(-1, -0.8, -0.5, 0.5, -0.3, 0) 
  shift_s2 <- c(1, 0.8, 0.6, -0.6, -0.4, -0.2)
  
  D_s1_raw <- generate_data(source_size, shift_s1)
  D_s1_xgb <- xgb.DMatrix(D_s1_raw$data, label = D_s1_raw$label)
  attr(D_s1_xgb, "censor") <- D_s1_raw$censor
  
  D_s2_raw <- generate_data(source_size, shift_s2)
  D_s2_xgb <- xgb.DMatrix(D_s2_raw$data, label = D_s2_raw$label)
  attr(D_s2_xgb, "censor") <- D_s2_raw$censor
  
  md_target <- xgb.train(param, D_target_train_xgb, nround = nround)
  md_s1     <- xgb.train(param, D_s1_xgb, nround = nround)
  md_s2     <- xgb.train(param, D_s2_xgb, nround = nround)
  
  all_mds <- list(md_target, md_s1, md_s2)
  weights <- sapply(all_mds, function(m) {
    p <- predict(m, D_target_train_xgb)
    concordance(Surv(D_target$label[idx_train], D_target$censor[idx_train]) ~ exp(p))$concordance[1]
  })
  w_norm <- weights / sum(weights)
  
  c_target <- concordance(test_surv ~ exp(predict(md_target, D_target_test_xgb)))$concordance[1]
  c_s1     <- concordance(test_surv ~ exp(predict(md_s1, D_target_test_xgb)))$concordance[1]
  c_s2     <- concordance(test_surv ~ exp(predict(md_s2, D_target_test_xgb)))$concordance[1]
  
  p_w <- w_norm[1]*predict(md_target, D_target_test_xgb) + 
    w_norm[2]*predict(md_s1, D_target_test_xgb) + 
    w_norm[3]*predict(md_s2, D_target_test_xgb)
  c_weighted <- concordance(test_surv ~ exp(p_w))$concordance[1]
  
  return(c(Target_Only = c_target, Source1_Only = c_s1, Source2_Only = c_s2, Weighted = c_weighted))
}

# --- 3. Parallel computing ---
num_cores <- 6
cl <- makeCluster(num_cores)
registerDoParallel(cl)

exp_grid <- expand.grid(target_size = TARGET_TOTAL_SIZES, source_size = SOURCE_SIZES, rep = 1:NUM_REPLICATIONS)

results_raw <- foreach(i = 1:nrow(exp_grid), .packages = c("xgboost", "survival"), .combine = "rbind") %dopar% {
  res <- run_compare_experiment(exp_grid$target_size[i], exp_grid$source_size[i], param, nround, TRAIN_TEST_RATIO)
  data.frame(Target_Size = exp_grid$target_size[i], Source_Size = exp_grid$source_size[i], 
             Method = names(res), CIndex = as.numeric(res))
}
stopCluster(cl)
results_raw<-read.csv("C:/Users/lenovo/Desktop/AFTBoost/c30/results_raw.csv")
summary_plot_data <- results_raw %>%
  group_by(Target_Size, Source_Size, Method) %>%
  summarise(Mean_C = mean(CIndex), SD_C = sd(CIndex), .groups = "drop")
write.csv(results_raw, file = "C:/Users/lenovo/Desktop/AFTBoost/c30/results_raw.csv", row.names = FALSE)# Convert multiple values into a vector
#results_raw<-read.csv("C:/Users/lenovo/Desktop/AFTBoost/c30/results_raw.csv")
summary_plot_data$Method <- factor(summary_plot_data$Method, 
                                   levels = c("Target_Only", "Source1_Only", "Source2_Only", "Weighted"),
                                   labels = c("Target Only", "S1 Only", "S2 Only", "Weighted Transfer"))
#write.csv(summary_plot_data, file = "C:/Users/lenovo/Desktop/AFTBoost/c30/summary_plot_data.csv", row.names = FALSE)# Convert multiple values into a vector
# --- 5. Plotting (rows and columns swapped) ---
#summary_plot_data<-read.csv("C:/Users/lenovo/Desktop/AFTBoost/c30/summary_plot_data.csv")
p_swapped <- ggplot(summary_plot_data, aes(x = factor(Target_Size), y = Mean_C, fill = Method)) +
  geom_bar(stat = "identity", width = 0.6, alpha = 0.85) +
  geom_errorbar(aes(ymin = Mean_C - SD_C, ymax = Mean_C + SD_C), 
                width = 0.2, color = "black") +
  # Trend line
  geom_line(aes(group = 1), color = "red", linetype = "dashed", alpha = 0.4) +
  geom_point(color = "red", shape = 17, size = 2) +
  
  # Key modification: facet rows by Source_Size and columns by Method
  facet_grid(Source_Size ~ Method, 
             labeller = labeller(
               Source_Size = function(x) paste("Source N =", x),
               Method = label_value
             )) +
  
  # Set lower y-axis limit to 0.55 to highlight differences
  coord_cartesian(ylim = c(0.55, NA)) + 
  
  scale_fill_manual(values = c("Target Only" = "#76c7b7", "S1 Only" = "#f59e7d", 
                               "S2 Only" = "#a3b7d9", "Weighted Transfer" = "#e6a2c3")) +
  
  labs(x = "Target N", 
       y = "C-index (Mean ± SD)", 
       title = "") +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "none", 
    strip.background = element_rect(fill = "gray95"),
    strip.text = element_text(face = "bold", size = 9),
    panel.spacing = unit(0.5, "lines"),
    panel.grid.minor = element_blank()
  )

print(p_swapped)

# Save file (variable name corrected)
ggsave("C:/Users/lenovo/Desktop/AFTBoost/c30/c30.png", 
       plot =p_swapped, width = 8, height = 8, dpi = 600)