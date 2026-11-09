#helper functions for bigram PCA

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
      filter(!is.na(term)) %>%
      separate(term, c("word1", "word2"), sep = " ") %>%
      filter(
        nchar(word1) > 2,
        nchar(word2) > 2,
        !word1 %in% tidytext::stop_words$word,
        !word2 %in% tidytext::stop_words$word
      ) %>%
      unite(term, word1, word2, sep = " ")
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