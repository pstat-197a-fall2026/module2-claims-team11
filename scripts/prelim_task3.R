# Preliminary Task 3: Neural Network

library(tidyverse)
library(rvest)
library(keras)
library(tensorflow)
library(tidymodels)
source("scripts/preprocessing.R")
# Load the data
load("data/claims-raw.RData")

glimpse(claims_raw)
#Cleaning webpage
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

claims_task3 <- claims_raw %>%
  parse_data()

set.seed(110826)

split3 <- initial_split(
  claims_task3,
  prop = 0.8,
  strata = bclass
)

train3 <- training(split3)
valid3 <- testing(split3)

cat("Training pages:", nrow(train3), "\n")
cat("Validation pages:", nrow(valid3), "\n")

#Prepare text for neural network
train_text3 <- train3$text_clean
valid_text3 <- valid3$text_clean

train_labels3 <- as.numeric(train3$bclass) - 1
valid_labels3 <- as.numeric(valid3$bclass) - 1

levels(train3$bclass)

length(train_text3)
length(valid_text3)

table(train_labels3)
table(valid_labels3)

#Converting to numerical 
preprocess_layer3 <- layer_text_vectorization(
  standardize = NULL,
  split = "whitespace",
  max_tokens = 5000,
  output_mode = "tf_idf"
)

preprocess_layer3 %>% adapt(train_text3)

#Neural network model

model3 <- keras_model_sequential() %>%
  preprocess_layer3() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 25, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model3)


#Training 

model3 %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "binary_accuracy"
)

# Train the neural network
history3 <- model3 %>%
  fit(
    train_text3,
    train_labels3,
    validation_data = list(valid_text3, valid_labels3),
    epochs = 5,
    batch_size = 32
  )

# Evaluating

valid_prob3 <- predict(model3, valid_text3) %>%
  as.numeric()

valid_pred3 <- as.integer(valid_prob3 >= 0.5)

accuracy3 <- mean(valid_pred3 == valid_labels3)

cat("Validation accuracy:", accuracy3, "\n")

table(
  Actual = valid_labels3,
  Predicted = valid_pred3
)

# Experiment 2: Larger neural network with more dropout

model3b <- keras_model_sequential() %>%
  preprocess_layer3() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")

model3b %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "binary_accuracy"
)

history3b <- model3b %>%
  fit(
    train_text3,
    train_labels3,
    validation_data = list(valid_text3, valid_labels3),
    epochs = 5,
    batch_size = 32
  )

# Evaluating
valid_prob3b <- as.numeric(predict(model3b, valid_text3))

accuracy3b <- mean(
  as.integer(valid_prob3b >= 0.5) == valid_labels3
)

cat("Model 1 accuracy:", accuracy3, "\n")
cat("Model 2 accuracy:", accuracy3b, "\n")

#Trained 2 neural networks to predict whether a webpage contained relevant claim content. Both models were trained for five rounds using the Adam optimizer and binary cross-entropy loss. Our first model used 25 neurons and a dropout rate of 0.2, achieving 81.1% accuracy. We then tested a model with 64 neurons and a dropout rate of 0.4, which improved accuracy slightly to 81.5%. Overall, our neural network performed better than the models from Task 2, which achieved 77.1% accuracy.
