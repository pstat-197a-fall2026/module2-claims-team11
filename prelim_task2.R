library(tidyverse)
library(tidytext)
library(rvest)

load("data/claims-raw.RData")

# Extract only meaningful webpage content
clean_html_task2 <- function(html) {
  tryCatch({
    
    read_html(html) %>%
      html_elements("title, h1, h2, h3, h4, h5, h6, p") %>%
      html_text2() %>%
      str_c(collapse = " ") %>%
      str_remove_all("https?://\\S+|www\\.\\S+") %>%
      str_replace_all("[[:punct:][:digit:]]", " ") %>%
      str_to_lower() %>%
      str_squish()
    
  }, error = function(e) "")
}

# Reprocess the original webpage data
claims_task2 <- claims_raw %>%
  filter(!is.na(text_tmp),
         str_detect(text_tmp, "<!"),
         !is.na(bclass)) %>%
  mutate(
    text_clean = map_chr(text_tmp, clean_html_task2)
  )

# Create bigrams
bigram_counts <- claims_task2 %>%
  select(.id, text_clean) %>%
  unnest_tokens(
    bigram,
    text_clean,
    token = "ngrams",
    n = 2
  ) %>%
  filter(!is.na(bigram)) %>%
  count(.id, bigram, name = "n")

# Find the 20 most common bigrams
bigram_counts %>%
  group_by(bigram) %>%
  summarise(total = sum(n), .groups = "drop") %>%
  arrange(desc(total)) %>%
  slice_head(n = 20)


# ============================================================
# TASK 2: LOGISTIC PCA REGRESSION WITH BIGRAMS
# ============================================================

library(tidyverse)
library(tidytext)

# Prepare the labeled data
task2_data <- claims_task2 %>%
  filter(!is.na(bclass), !is.na(text_clean),
         text_clean != "") %>%
  distinct(.id, .keep_all = TRUE) %>%
  mutate(bclass = factor(bclass))

stopifnot(nlevels(task2_data$bclass) == 2)

# 80/20 split, stratified by class
set.seed(197)

train_idx <- unlist(
  lapply(split(seq_len(nrow(task2_data)),
               task2_data$bclass), function(i) {
                 sample(i, floor(0.8 * length(i)))
               }),
  use.names = FALSE
)

train <- task2_data[train_idx, ]
valid <- task2_data[-train_idx, ]

positive_class <- levels(task2_data$bclass)[2]

train$y <- as.integer(train$bclass == positive_class)
valid$y <- as.integer(valid$bclass == positive_class)

cat("Training pages:", nrow(train), "\n")
cat("Validation pages:", nrow(valid), "\n")


# Helper 1: Tokenize text into words or bigrams

get_counts <- function(text, type) {
  
  df <- tibble(
    doc_id = seq_along(text),
    text = text
  )
  
  if (type == "word") {
    
    df <- df %>%
      unnest_tokens(term, text, token = "words") %>%
      filter(
        nchar(term) > 2,
        !term %in% tidytext::stop_words$word
      )
    
  } else {
    
    df <- df %>%
      unnest_tokens(
        term, text,
        token = "ngrams", n = 2
      ) %>%
      filter(
        !is.na(term),
        !str_detect(term, "\\b[a-z]\\b")
      )
  }
  
  df %>%
    count(doc_id, term, name = "n")
}


# Helper 2: Create TF-IDF matrices and PCA scores

make_pca <- function(train_text, test_text,
                     type, n_pc = 20,
                     max_terms = 400) {
  
  train_counts <- get_counts(train_text, type)
  test_counts <- get_counts(test_text, type)
  
  # Choose vocabulary using training data only
  vocab <- train_counts %>%
    group_by(term) %>%
    summarise(
      doc_freq = n(),
      total = sum(n),
      .groups = "drop"
    ) %>%
    filter(
      doc_freq >= 3,
      doc_freq < 0.9 * length(train_text)
    ) %>%
    arrange(desc(total), term) %>%
    slice_head(n = max_terms) %>%
    mutate(
      term_id = row_number(),
      idf = log(
        (length(train_text) + 1) /
          (doc_freq + 1)
      ) + 1
    )
  
  if (nrow(vocab) < 2) {
    stop("Not enough terms for PCA.")
  }
  
  # Convert counts to TF-IDF matrices
  make_matrix <- function(counts, n_docs) {
    
    counts <- counts %>%
      group_by(doc_id) %>%
      mutate(tf = n / sum(n)) %>%
      ungroup() %>%
      inner_join(
        vocab %>% select(term, term_id, idf),
        by = "term"
      )
    
    mat <- matrix(
      0,
      nrow = n_docs,
      ncol = nrow(vocab)
    )
    
    if (nrow(counts) > 0) {
      mat[cbind(counts$doc_id, counts$term_id)] <-
        counts$tf * counts$idf
    }
    
    colnames(mat) <- vocab$term
    mat
  }
  
  x_train <- make_matrix(
    train_counts, length(train_text)
  )
  
  x_test <- make_matrix(
    test_counts, length(test_text)
  )
  
  k <- min(
    n_pc, nrow(x_train) - 1,
    ncol(x_train) - 1
  )
  
  # PCA, fitted on training data only
  pca <- prcomp(
    x_train,
    center = TRUE,
    scale. = FALSE,
    rank. = k
  )
  
  train_pc <- as.data.frame(
    pca$x[, seq_len(k), drop = FALSE]
  )
  
  test_pc <- as.data.frame(
    predict(pca, newdata = x_test)[
      , seq_len(k), drop = FALSE
    ]
  )
  
  names(train_pc) <- paste0("PC", seq_len(k))
  names(test_pc) <- paste0("PC", seq_len(k))
  
  list(train = train_pc, test = test_pc)
}


# Helper 3: Logistic principal component regression

fit_word_pcr <- function(train_text, train_y, test_text) {
  
  pcs <- make_pca(
    train_text, test_text, type = "word"
  )
  
  model_data <- data.frame(
    y = train_y,
    pcs$train
  )
  
  model <- glm(
    y ~ .,
    data = model_data,
    family = binomial()
  )
  
  # Return predicted log-odds
  as.numeric(
    predict(
      model,
      newdata = pcs$test,
      type = "link"
    )
  )
}



# Model 1: Word-only logistic PCA regression
# ------------------------------------------------------------

cat("\nFitting word-only model...\n")

word_valid_logodds <- fit_word_pcr(
  train$text_clean,
  train$y,
  valid$text_clean
)

word_valid_prob <- plogis(word_valid_logodds)

word_accuracy <- mean(
  (word_valid_prob >= 0.5) == valid$y
)


# ------------------------------------------------------------
# Create out-of-fold word-model predictions
# ------------------------------------------------------------

# These help avoid overfitting the second-stage model.
set.seed(197)
fold_id <- sample(
  rep(1:5, length.out = nrow(train))
)

word_oof_logodds <- rep(NA_real_, nrow(train))

for (fold in 1:5) {
  
  cat("Word-model fold:", fold, "\n")
  
  fit_rows <- which(fold_id != fold)
  holdout_rows <- which(fold_id == fold)
  
  word_oof_logodds[holdout_rows] <- fit_word_pcr(
    train$text_clean[fit_rows],
    train$y[fit_rows],
    train$text_clean[holdout_rows]
  )
}



# Model 2: Add bigram principal components
# ------------------------------------------------------------

cat("\nCalculating bigram principal components...\n")

bigram_pcs <- make_pca(
  train$text_clean,
  valid$text_clean,
  type = "bigram"
)

# Second-stage training data
stage2_train <- data.frame(
  y = train$y,
  word_logodds = word_oof_logodds,
  bigram_pcs$train
)

# Second-stage validation data
stage2_valid <- data.frame(
  word_logodds = word_valid_logodds,
  bigram_pcs$test
)

# Comparator using word log-odds only
calibrated_word_model <- glm(
  y ~ word_logodds,
  data = stage2_train,
  family = binomial()
)

calibrated_prob <- predict(
  calibrated_word_model,
  newdata = stage2_valid,
  type = "response"
)

# Word log-odds + bigram PCs
combined_model <- glm(
  y ~ .,
  data = stage2_train,
  family = binomial()
)

combined_prob <- predict(
  combined_model,
  newdata = stage2_valid,
  type = "response"
)

# Compare validation accuracies
results_task2 <- tibble(
  Model = c(
    "Word-only logistic PCA",
    "Word log-odds only (stage 2)",
    "Word log-odds + bigram PCs"
  ),
  Accuracy = c(
    word_accuracy,
    mean((calibrated_prob >= 0.5) == valid$y),
    mean((combined_prob >= 0.5) == valid$y)
  )
)

print(results_task2)

# Improvement attributable to bigram features
bigram_improvement <- 100 * (
  results_task2$Accuracy[3] -
    results_task2$Accuracy[2]
)

cat(
  "\nBigram accuracy change:",
  round(bigram_improvement, 2),
  "percentage points\n"
)

# Save results
dir.create("results", showWarnings = FALSE)

write.csv(
  results_task2,
  "results/task2-accuracy-comparison.csv",
  row.names = FALSE
)



# TASK 2: MODEL DIAGNOSTICS
# ============================================================

# 1. Check actual class proportions
cat("\nActual validation class proportions:\n")
print(prop.table(table(valid$bclass)))

# 2. Majority-class baseline accuracy
majority_accuracy <- max(
  prop.table(table(valid$y))
)

cat("\nMajority-class baseline accuracy:",
    round(majority_accuracy * 100, 2), "%\n")

# 3. Confusion matrix: word-only model
cat("\nWord-only confusion matrix:\n")
print(table(
  Actual = valid$y,
  Predicted = as.integer(word_valid_prob >= 0.5)
))

# 4. Confusion matrix: word + bigrams
cat("\nWord + bigrams confusion matrix:\n")
print(table(
  Actual = valid$y,
  Predicted = as.integer(combined_prob >= 0.5)
))

# 5. Check predicted probabilities
cat("\nWord-only probabilities:\n")
print(summary(word_valid_prob))

cat("\nWord + bigram probabilities:\n")
print(summary(combined_prob))

# 6. Compare whether the models predict different classes
cat("\nNumber of predictions changed by adding bigrams:\n")
print(sum(
  (word_valid_prob >= 0.5) !=
    (combined_prob >= 0.5)
))
