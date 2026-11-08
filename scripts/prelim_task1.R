library(tidyverse)
library(tidymodels)
library(tidytext)
library(rvest)
library(qdapRegex)

# Loading helper functions and raw data
source("scripts/preprocessing.R")
source("scripts/bigram_pca_helper.R")
load("data/claims-raw.RData")

# Defining baseline parser for paragraphs only
parse_fn_p <- function(.html) {
  read_html(.html) %>%
    html_elements("p") %>%
    html_text2() %>%
    str_c(collapse = " ") %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all("'") %>%
    str_replace_all(paste(c("\n", "[[:punct:]]", "nbsp", "[[:digit:]]", "[[:symbol:]]"),
                          collapse = "|"), " ") %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# Defining parser for head metadata and paragraphs
parse_fn_head <- function(.html) {
  read_html(.html) %>%
    html_elements("head, p") %>%
    html_text2() %>%
    str_c(collapse = " ") %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all("'") %>%
    str_replace_all(paste(c("\n", "[[:punct:]]", "nbsp", "[[:digit:]]", "[[:symbol:]]"),
                          collapse = "|"), " ") %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# Filtering valid HTML rows once for consistent comparisons
claims_valid <- claims_raw %>%
  filter(str_detect(text_tmp, "<!"))

# Parsing baseline data with paragraphs only
claims_p <- claims_valid %>%
  rowwise() %>%
  mutate(text_clean = parse_fn_p(text_tmp)) %>%
  unnest(text_clean)

# Parsing data with head and paragraphs
claims_head <- claims_valid %>%
  rowwise() %>%
  mutate(text_clean = parse_fn_head(text_tmp)) %>%
  unnest(text_clean)

# Parsing augmented data with h1-h6 and paragraphs using parse_data from preprocessing.R
claims_headers <- claims_raw %>%
  parse_data()

# Defining function for fitting and evaluating logistic PCR
evaluate_logistic_pcr <- function(clean_df, label_name, n_pc = 20, max_terms = 400, seed = 110826) {
  
  # Splitting data into training and testing sets
  set.seed(seed)
  split <- initial_split(clean_df, prop = 0.8, strata = bclass)
  train_data <- training(split)
  test_data  <- testing(split)
  
  # Creating 5 stratified folds for cross-validation
  set.seed(seed)
  folds <- vfold_cv(train_data, v = 5, strata = bclass)
  cv_preds <- vector("list", 5)
  
  # Running 5-fold cross-validation loop
  for (i in seq_along(folds$splits)) {
    fold_train <- analysis(folds$splits[[i]])
    fold_test  <- assessment(folds$splits[[i]])
    
    # Computing word PCA features
    pca_fold <- make_pca(
      train_text = fold_train$text_clean,
      test_text  = fold_test$text_clean,
      type       = "word",
      n_pc       = n_pc,
      max_terms  = max_terms
    )
    
    # Fitting logistic regression on principal components
    train_df_glm <- pca_fold$train %>%
      mutate(bclass = fold_train$bclass)
    
    glm_fit <- glm(bclass ~ ., data = train_df_glm, family = "binomial")
    
    # Predicting probabilities on validation fold
    probs <- predict(glm_fit, newdata = pca_fold$test, type = "response")
    
    cv_preds[[i]] <- tibble(
      truth = fold_test$bclass,
      prob  = as.numeric(probs)
    )
  }
  
  # Summarizing cross-validation metrics
  cv_summary <- bind_rows(cv_preds) %>%
    mutate(
      pred = factor(
        if_else(prob >= 0.5, levels(train_data$bclass)[2], levels(train_data$bclass)[1]),
        levels = levels(train_data$bclass)
      )
    ) %>%
    summarise(
      split       = "5-Fold CV",
      model       = label_name,
      accuracy    = accuracy_vec(truth, pred),
      sensitivity = sens_vec(truth, pred, event_level = "second"),
      specificity = spec_vec(truth, pred, event_level = "second"),
      roc_auc     = roc_auc_vec(truth, prob, event_level = "second")
    )
  
  # Computing PCA features on full training set
  pca_full <- make_pca(
    train_text = train_data$text_clean,
    test_text  = test_data$text_clean,
    type       = "word",
    n_pc       = n_pc,
    max_terms  = max_terms
  )
  
  # Fitting final logistic regression model
  train_full_glm <- pca_full$train %>%
    mutate(bclass = train_data$bclass)
  
  glm_final <- glm(bclass ~ ., data = train_full_glm, family = "binomial")
  
  # Generating predictions on holdout test set
  test_probs <- predict(glm_final, newdata = pca_full$test, type = "response")
  
  # Summarizing test set metrics
  test_summary <- tibble(
    truth = test_data$bclass,
    prob  = as.numeric(test_probs)
  ) %>%
    mutate(
      pred = factor(
        if_else(prob >= 0.5, levels(train_data$bclass)[2], levels(train_data$bclass)[1]),
        levels = levels(train_data$bclass)
      )
    ) %>%
    summarise(
      split       = "Test Set",
      model       = label_name,
      accuracy    = accuracy_vec(truth, pred),
      sensitivity = sens_vec(truth, pred, event_level = "second"),
      specificity = spec_vec(truth, pred, event_level = "second"),
      roc_auc     = roc_auc_vec(truth, prob, event_level = "second")
    )
  
  bind_rows(cv_summary, test_summary)
}

# Evaluating baseline model with paragraphs only
res_p <- evaluate_logistic_pcr(
  claims_p,
  label_name = "Paragraphs Only (p)",
  n_pc = 20,
  max_terms = 400
)

# Evaluating model with head metadata and paragraphs
res_head <- evaluate_logistic_pcr(
  claims_head,
  label_name = "Head + Paragraphs (head, p)",
  n_pc = 20,
  max_terms = 400
)

# Evaluating augmented model with headings and paragraphs
res_headers <- evaluate_logistic_pcr(
  claims_headers,
  label_name = "Headings + Paragraphs (h1-h6, p)",
  n_pc = 20,
  max_terms = 400
)

# Combining and printing comparison results across all three strategies
final_comparison <- bind_rows(res_p, res_head, res_headers) %>%
  arrange(split, desc(roc_auc))

print(final_comparison)