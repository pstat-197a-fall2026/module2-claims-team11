library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)
library(tidytext)
library(rvest)
source("scripts/extra_ngram_pca_helpers.R") # TFIDF Helper Functions
source("scripts/preprocessing.R") # Preprocessing Functions
load("data/claims-test.RData") # Data to Predict
load("data/claims-test-cleaned.RData") # Preprocessed Data
load("data/binary_idf_levels.rda") # Binary levels and IDF values from training
set.seed(1)
model <- load_model_tf("results/binary_nn") # Binary Neural Network

# Preprocess raw HTML
# claims_clean_test <- claims_test %>% parse_data()

# Convert to TFIDF matrix for NN input
tokens <- claims_clean_test %>% tokenize_fn(n = 1)
tf <- tokens %>% tf_fn()
tfidf <- tfidf_fn_updated(tf, idf, claims_clean_test) %>% select(-.id) %>% as.matrix()

# Generate predictions
predictions <- predict(model, tfidf)
probabilities <- as.numeric(predictions)
predicted_class <- factor(if_else(probabilities >= 0.5, class_levels[2], class_levels[1]), levels = class_levels)
binary_preds <- data.frame(.id = claims_clean_test$.id, bclass.pred = predicted_class)

save(binary_preds, file = "results/binary_predictions.rda")