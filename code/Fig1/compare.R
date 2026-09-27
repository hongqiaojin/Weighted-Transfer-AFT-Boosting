# Load necessary libraries
library(xgboost)
library(survival)
library(randomForestSRC)
library(foreach)
library(doParallel)
library(ggplot2)
library(tidyr)
library(dplyr)

# ================= 1. Define custom loss functions =================

# AFTBoost (normal distribution-based AFT model)
myobject <- function(preds, dtrain) {
  T <- getinfo(dtrain, "label")
  d <- attr(dtrain, 'censor')
  a <- 1
  u_val <- (log(T) - preds) / a
  fai <- dnorm(u_val)
  Fai <- pnorm(u_val)
  
  # Avoid division by zero using a small threshold
  g <- fai / pmax(1 - Fai, 1e-7)
  grad <- -(d * u_val) - g * (1 - d)
  
  # Approximation of the second derivative (Hessian)
  h_term <- (fai * (u_val * (1 - Fai) - fai)) / pmax((1 - Fai)^2, 1e-7)
  hess <- d + ((1 - d) * h_term)
  hess <- pmax(hess, 1e-4) # Ensure a positive Hessian
  return(list(grad = grad, hess = hess))
}

evalerror <- function(preds, dtrain) {
  T <- getinfo(dtrain, "label")
  d <- attr(dtrain, "censor")
  u_val <- (log(T) - preds)
  fai <- dnorm(u_val)
  Fai <- pnorm(u_val)
  err <- -sum(d * log(pmax(fai/T, 1e-7)) + (1 - d) * log(pmax(1 - Fai, 1e-7)))
  return(list(metric = "deviance", value = err))
}

# SurvivalBoost (based on Cox partial likelihood)
mylossobj2 <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  censor <- attr(dtrain, "censor")
  ord <- order(labels)
  ran <- rank(labels)
  d <- censor[ord]
  haz <- exp(preds[ord])
  rsk <- rev(cumsum(rev(haz)))
  P <- outer(haz, rsk, '/')
  P[upper.tri(P)] <- 0
  grad <- -(d - P %*% d)
  H <- P - (outer(haz^2, rsk^2, '/'))
  H[upper.tri(H)] <- 0
  hess <- pmax(H %*% d, 1e-4)
  return(list(grad = grad[ran], hess = hess[ran]))
}

evalerror2 <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  censor <- attr(dtrain, "censor")
  ord <- order(labels)
  d <- censor[ord]
  haz <- exp(preds[ord])
  rsk <- rev(cumsum(rev(haz)))
  err <- -sum(d * (preds[ord] - log(pmax(rsk, 1e-7))))
  return(list(metric = "deviance", value = err))
}

# ================= 2. Wrap the core simulation function =================

run_simulation <- function(n, u, iter) {
  set.seed(123 + iter + n) # Dynamic seed to ensure independence
  
  # Data generation
  x <- matrix(rnorm(n * 6), n, 6)
  colnames(x) <- paste0("x", 1:6)
  v1 <- rnorm(n, 0, 1)
  
  # Generate nonlinear survival times
  real.time <- exp(sin(x[,1] + 1) + x[,2] * x[,3] - x[,4]^2 - 0.3 * abs(x[,5]) - 0.2 * x[,6] + v1)
  cens.time <- runif(n, min = 0, max = u)
  status <- ifelse(real.time <= cens.time, 1, 0)
  obs.time <- ifelse(real.time <= cens.time, real.time, cens.time)
  
  t_split <- round(n * 0.6)
  train_idx <- 1:t_split
  test_idx <- (t_split + 1):n
  
  # Evaluation target
  test_surv <- Surv(obs.time[test_idx], status[test_idx])
  
  # --- Model 1: AFTBoost ---
  dtrain <- xgb.DMatrix(x[train_idx,], label = obs.time[train_idx])
  attr(dtrain, "censor") <- status[train_idx]
  dtest <- xgb.DMatrix(x[test_idx,])
  md1 <- xgb.train(list(max_depth=3, eta=0.035, objective=myobject, eval_metric=evalerror), 
                   dtrain, nround=200, verbose=0)
  c1 <- concordance(test_surv ~ exp(predict(md1, dtest)))$concordance
  
  # --- Model 2: SurvivalBoost ---
  md2 <- xgb.train(list(max_depth=3, eta=0.035, objective=mylossobj2, eval_metric=evalerror2), 
                   dtrain, nround=200, verbose=0)
  c2 <- 1 - concordance(test_surv ~ predict(md2, dtest))$concordance
  
  # --- Model 3: RSF ---
  df_train <- as.data.frame(cbind(x[train_idx,], time=obs.time[train_idx], status=status[train_idx]))
  rf_fit <- rfsrc(Surv(time, status) ~ ., data = df_train, ntree = 100, nsplit = 5)
  rf_pred <- predict(rf_fit, newdata = as.data.frame(x[test_idx,]))
  c3 <- concordance(test_surv ~ rowMeans(rf_pred$survival))$concordance
  
  # --- Model 4: Cox ---
  cox_fit <- tryCatch(coxph(Surv(time, status) ~ ., data = df_train), error = function(e) return(NULL))
  if(!is.null(cox_fit)) {
    c4 <- 1 - concordance(test_surv ~ predict(cox_fit, newdata = as.data.frame(x[test_idx,]), type="risk"))$concordance
  } else { c4 <- NA }
  
  # --- Model 5: AFT ---
  aft_fit <- tryCatch(survreg(Surv(time, status) ~ ., data = df_train, dist = "lognormal"), error = function(e) return(NULL))
  if(!is.null(aft_fit)) {
    c5 <- concordance(test_surv ~ predict(aft_fit, newdata = as.data.frame(x[test_idx,])))$concordance
  } else { c5 <- NA }
  
  return(c(AFTBoost=c1, SurvivalBoost=c2, RSF=c3, Cox=c4, AFT=c5))
}

# ================= 3. Run simulations in parallel =================

n_vals <- c(100, 300, 500)
u_vals <- c(2, 4)
reps <- 500 # Recommended to increase to 50+ for more stable boxplots

# Start parallel computing
cl <- makeCluster(10)
registerDoParallel(cl)

all_results <- list()

for(u in u_vals) {
  for(n in n_vals) {
    cat(sprintf("Processing: n=%d, u=%d\n", n, u))
    res <- foreach(i = 1:reps, .combine = rbind, 
                   .packages = c("xgboost", "survival", "randomForestSRC")) %dopar% {
                     run_simulation(n, u, i)
                   }
    df_res <- as.data.frame(res)
    df_res$n <- n
    df_res$u <- u
    all_results[[paste(n, u, sep="_")]] <- df_res
  }
}

stopCluster(cl)

# ================= 4. Data organization and plotting =================
# ================= 4. Data organization and plotting (with whisker caps added) =================

# 1. Merge data and rename labels
final_df <- bind_rows(all_results) %>%
  mutate(
    # According to your requirement: map u=2 to 50% and u=4 to 30%
    Censoring = ifelse(u == 2, "Censoring = 50%", "Censoring = 30%"),
    SampleSize = factor(paste0("n = ", n), levels = c("n = 100", "n = 300", "n = 500"))
  ) %>%
  pivot_longer(cols = AFTBoost:AFT, names_to = "Method", values_to = "C_Index")
data_last<-as.data.frame(final_df)
write.csv(data_last, file = "C:/Users/lenovo/Desktop/AFTBoost/box_aft.csv", row.names = FALSE)# Convert multiple values into a vector
# 2. Set the display order of methods
final_df$Method <- factor(final_df$Method, 
                          levels = c("AFTBoost", "SurvivalBoost", "RSF", "Cox", "AFT"))

# 3. Plot
p<-ggplot(final_df, aes(x = Method, y = C_Index, fill = Method)) +
  # --- Key modification: add horizontal caps at the ends of whiskers ---
  stat_boxplot(geom = "errorbar", width = 0.3) + 
  # --------------------------------------
geom_boxplot(outlier.size = 0.5, alpha = 0.8) +
  facet_grid(Censoring ~ SampleSize) +
  theme_bw() +
  labs(title = "",
       y = "C-index", 
       x = "") +
  scale_fill_brewer(palette = "Set2") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    strip.text = element_text(size = 11, face = "bold"), # Panel label font
    strip.background = element_rect(fill = "#f0f0f0"),
    legend.position = "none",
    panel.grid.minor = element_blank() # Remove minor background grid lines
  )
#print(p_optimized)
ggsave(file = "C:/Users/lenovo/Desktop/AFTBoost/box_aft.png", plot =p, dpi = 800,width = 7, height = 5 )