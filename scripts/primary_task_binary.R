library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)
library(tidytext)
library(rvest)

#rm(list = ls())
source("scripts/preprocessing_reference.R")
source("scripts/bigram_pca_helper.R")
load('data/claims-clean-example.RData')

### Binary Classification models:

## Binary NN with Bigram PCA, Adamw optimization, 5-fold CV

# Initial 80/20 split
set.seed(197)

b_partitions <- claims_clean |>
  initial_split(prop = 0.8, strata = bclass)

b_train_data <- training(b_partitions)
b_test_data  <- testing(b_partitions)

# text
b_train <- b_train_data$text_clean
b_test <- b_test_data$text_clean

# labels
b_train_labels <- b_train_data$bclass |> as.numeric() - 1
b_test_labels <- b_test_data$bclass

# Create 5 stratified folds
set.seed(197)

b_fold <- vfold_cv(
  data = b_train_data,
  v = 5,
  strata = bclass
)

# Cross-validation loop process
b_cv_results <- vector("list", 5)

for (i in seq_along(b_fold$splits)) {
  
  fold_train <- analysis(b_fold$splits[[i]])
  fold_test <- assessment(b_fold$splits[[i]])
  
  # Extract text
  train_text <- fold_train$text_clean
  test_text <- fold_test$text_clean
  
  # Extract labels
  train_y <- fold_train$bclass |> as.numeric() - 1
  test_y <- fold_test$bclass
  
  # Bigram PCA with make_pca helper function
  pca_fold <- make_pca(
    train_text = train_text,
    test_text = test_text,
    type = "bigram",
    n_pc = 20,
    max_terms = 400
  )
  
  # pca projected features
  X_train <- as.matrix(pca_fold$train)
  X_test <- as.matrix(pca_fold$test)
  
  # Neural network specs, 1 hidden layer with 64 neurons
  # Adamw optimization, and using logit for backpropogation
  b_model <- keras_model_sequential(
    input_shape = c(ncol(X_train))) |>
    layer_dense(units = 64, activation = "relu") |>
    layer_dropout(rate = 0.4) |>
    layer_dense(units = 1)
  
  b_model |> compile(
    loss = loss_binary_crossentropy(from_logits = T),
    optimizer = tf$keras$optimizers$AdamW(
      learning_rate = 0.001,
      weight_decay = 0.001,
      beta_1 = 0.9,
      beta_2 = 0.999)
    )
  
  # Train
  b_model |> fit(
    x = X_train,
    y = train_y,
    epochs = 50,
    batch_size = 32,
    verbose = 0
  )
  
  # Predict
  logits <- b_model |> predict(X_test, verbose = 0)
  
  probabilities <- plogis(as.numeric(logits))
  
  b_cv_results[[i]] <- tibble(
    fold = i,
    truth = test_y,
    probability = probabilities
  )
  
  cat("Completed fold:", i, "\n")
}

b_cv_predictions <- bind_rows(b_cv_results)

# Metric set for cv
b_cv_metrics <- b_cv_predictions |>
  mutate(
    predicted = factor(
      if_else(
        probability >= 0.5,
        levels(b_train_data$bclass)[2],
        levels(b_train_data$bclass)[1]
      ),
      levels = levels(b_train_data$bclass)
    )
  ) |>
  summarise(
    accuracy = accuracy_vec(truth, predicted),
    sensitivity = sens_vec(
      truth, predicted,
      event_level = "second"
    ),
    specificity = spec_vec(
      truth, predicted,
      event_level = "second"
    ),
    roc_auc = roc_auc_vec(
      truth, probability,
      event_level = "second"
    )
  )

b_cv_metrics

# accuracy sensitivity specificity roc_auc
#  0.689       0.536       0.858   0.740

# Full training model PCA
bigram_pcs <- make_pca(
  train_text = b_train,
  test_text = b_test,
  type = "bigram",
  n_pc = 20,
  max_terms = 400
)

# pca projected features
bpca_train <- as.matrix(bigram_pcs$train)
bpca_test <- as.matrix(bigram_pcs$test)

# Final model
b_final_model <- keras_model_sequential(
  input_shape = c(ncol(bpca_train))) |>
  layer_dense(units = 64, activation = "relu") |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 1)

b_final_model |> compile(
  loss = loss_binary_crossentropy(from_logits = T),
  optimizer = tf$keras$optimizers$AdamW(
    learning_rate = 0.001,
    weight_decay = 0.001,
    beta_1 = 0.9,
    beta_2 = 0.999)
)

b_history <- b_final_model |> 
  fit(
  x = bpca_train,
  y = b_train_labels,
  epochs = 50,
  batch_size = 32,
  verbose = 1
)

# Test results
test_logits <- b_final_model |>
  predict(bpca_test, verbose = 0)

test_prob <- plogis(as.numeric(test_logits))

test_pred <- factor(
  if_else(
    test_prob >= 0.5,
    levels(b_train_data$bclass)[2],
    levels(b_train_data$bclass)[1]
  ),
  levels = levels(b_train_data$bclass)
)

b_test_metrics <- tibble(
  truth = b_test_labels,
  estimate = test_pred,
  probability = test_prob
) |>
  summarise(
    accuracy = accuracy_vec(truth, estimate),
    sensitivity = sens_vec(
      truth, estimate,
      event_level = "second"
    ),
    specificity = spec_vec(
      truth, estimate,
      event_level = "second"
    ),
    roc_auc = roc_auc_vec(
      truth, probability,
      event_level = "second"
    )
  )

b_test_metrics
#  accuracy sensitivity specificity roc_auc
#   0.669       0.498       0.858   0.695

# save the entire model as a SavedModel
save_model_tf(b_final_model, "results/binary-model1")

# Compared to preliminary task models, this PCA NN model did worse
# From CV we received .689 acc, .74 auc, and test .669 acc, .695 auc.
# Uses only bigram features, try word logit + bigram feature for future model
