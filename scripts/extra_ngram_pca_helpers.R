require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

# tokenizes parsed text into n-grams. Specify n. Output is the document id, classification labels, and each n-gram on a row
tokenize_fn <- function(.df, n = 1){
  if (n < 1 || n != as.integer(n)) {
    stop("n must be a positive integer.")
  }
  out <- .df %>%
    select(.id, text_clean) %>%
    unnest_tokens(output = word, input = text_clean, token = "words", to_lower = TRUE) %>%
    anti_join(stop_words, by = c("word" = "word")) %>%
    mutate(word = lemmatize_words(word)) %>%
    filter(str_length(word) > 2)
  if (n == 1){
    out <- out %>%
      mutate(term = word)
  }else{
    out <- out %>%
      group_by(.id) %>%
      summarise(term = list(tokenize_ngrams(word, n = n, n_min = n, simplify = TRUE)[[1]]), .groups = "drop") %>%
      unnest(term)
  }
  out %>% select(.id, term)
}


# generates term frequencies of each token in each text. The input here is the output from tokenize_fn
tf_fn <- function(token_data){
  out <- token_data %>%
    count(.id, term, name = "n") %>%
    group_by(.id) %>%
    mutate(tf = n / sum(n)) %>%
    ungroup()
  return(out)
}


# calculates idf values for each token, filtered by tokens with a minimum document frequency and maximum document proportion
# max_terms limits the number of idf values returned. Also determines the vocabulary from the tokens listed in tf_fn output
idf_fn <- function(tf_data, min_doc_freq = 3, max_doc_prop = 0.9, max_terms = 400){
  n_docs <- n_distinct(tf_data$.id)
  out <- tf_data %>%
    group_by(term) %>%
    summarise(doc_freq = n_distinct(.id), total = sum(n), .groups = "drop") %>%
    filter(doc_freq >= min_doc_freq, doc_freq < max_doc_prop * n_docs) %>%
    arrange(desc(total), term) %>%
    slice_head(n = max_terms) %>%
    mutate(idf = log((n_docs + 1)/(doc_freq + 1)) + 1) %>%
    select(term, doc_freq, idf)
  
  return(out)
}

# Generate a tf-idf matrix based on idf_fn and tf_fn outputs
# only calculates based on terms in the idf output
tfidf_fn <- function(tf_data, idf_data){
  # Get alphabetized vocabulary
  vocab <- idf_data %>%
    arrange(term) %>%
    pull(term)
  # Generate tf-idf values
  out <- tf_data %>%
    inner_join(idf_data %>% select(term, idf), by = "term") %>%
    mutate(tf_idf = tf * idf) %>%
    select(.id, term, tf_idf) %>%
    pivot_wider(id_cols = .id, names_from = term, values_from = tf_idf, values_fill = 0)
  
  # Make sure all analysis terms are in the tfidf table
  missing_terms <- setdiff(vocab, names(out))
  out[missing_terms] <- 0
  
  out <- out %>% select(.id, all_of(vocab)) %>% select(.id, sort(vocab))
  return(out)
}

# Train a PCA model using tfidf matrix
# specify either var_keep (minimum cumulative variance) or n_pc (fixed number of components)
pca_train_fn <- function(tfidf_data, var_keep = NULL, n_pc = NULL){
  if (is.null(var_keep) && is.null(n_pc)) {
    var_keep <- 0.80
  }
  if (!is.null(var_keep) && !is.null(n_pc)) {
    stop("Specify either var_keep or n_pc, not both.")
  }
  if (!is.null(var_keep) && (var_keep <= 0 || var_keep > 1)) {
    stop("var_keep must be greater than 0 and less than or equal to 1.")
  }
  x <- tfidf_data %>%
    select(-.id) %>%
    as.matrix()
  
  # Determine maximum possible components
  k_max <- min(nrow(x) - 1, ncol(x))
  if (k_max < 1) {
    stop("Not enough observations or features for PCA.")
  }
  if (!is.null(n_pc)) {
    if (n_pc < 1 || n_pc != as.integer(n_pc)) {
      stop("n_pc must be a positive integer.")
    }
    if (n_pc > k_max) {
      stop("n_pc exceeds the maximum number of available components.")
    }
  }
  pca_model <- prcomp(x, center = TRUE, scale. = FALSE, rank. = k_max)
  if (!is.null(var_keep)){
    var_explained <- pca_model$sdev^2 / sum(pca_model$sdev^2)
    cum_var <- cumsum(var_explained)
    n_pc <- which(cum_var >= var_keep)[1]
  }
  out <- list(model = pca_model, n_pc = n_pc)
  return(out)
}

# generate principle components given pca model and tfidf table
pca_predict_fn <- function(tfidf_data, pca_data){
  # Select TF-IDF feature columns
  x <- tfidf_data %>%
    select(-.id) %>%
    as.matrix()
  # Apply trained PCA model
  pcs <- predict(pca_data$model, newdata = x)
  # Select retained principal components
  out <- as.data.frame(pcs[, seq_len(pca_data$n_pc), drop = FALSE])
  # Add document identifiers
  out <- bind_cols(tfidf_data %>% select(.id), out)
  return(out)
}
