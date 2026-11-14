load("results/binary_predictions.rda")
load("results/multiclass_predictions.rda")

if(nrow(binary_preds) != 929 | nrow(multiclass_preds) != 929){
  stop("Missing some predictions")
}

pred_df <- inner_join(binary_preds, multiclass_preds, by = ".id")
save(pred_df, file = "results/preds-group11.RData")