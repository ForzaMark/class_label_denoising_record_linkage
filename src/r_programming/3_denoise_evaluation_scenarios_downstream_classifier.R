setwd("~/label_denoising_record_linkage/src/r_programming")
library(dplyr)

source("denoise.R")
source("extract_dataset_from_command_line_input.R")

args <- commandArgs(trailingOnly = TRUE)
result <- extract_dataset_from_command_line_input(args)

general_dataset <- result$general_dataset
dataset <- result$dataset

datasets <- list(
  c(glue("{dataset}_rf"), glue('../../datasets/evaluation_scenarios/experiment_down_stream_classifier/{dataset}_rf_corrupted_train_data.csv')),
  c(glue("{dataset}_svm"), glue('../../datasets/evaluation_scenarios/experiment_down_stream_classifier/{dataset}_svm_corrupted_train_data.csv')),
  c(glue("{dataset}_tree"), glue('../../datasets/evaluation_scenarios/experiment_down_stream_classifier/{dataset}_tree_corrupted_train_data.csv'))
)

detectors <- list(
  "cvcf",
  "harf"
)

for (detector_name in detectors) {
  results <- list()
  for (dataset_triple in datasets) {
    dataset_name <- dataset_triple[1]
    file <- dataset_triple[2]
    
    detector = ifelse(detector_name == 'harf', HARF, CVCF)
    
    print(detector_name)
    print(dataset_name)
    
    noisy_data <- read.csv(file)
    features = exclude_label_from_features(noisy_data)
    
    data <- noisy_data
    data$label <- as.factor(data$label)
    
    out <- detector(label~., data = data)
    
    
    if (length(out$remIdx) > 0) {
      cleaned_data <- noisy_data[-out$remIdx, ]
    } else {
      cleaned_data = noisy_data
    }

    write.csv(cleaned_data, glue("../../datasets/evaluation_scenarios/experiment_down_stream_classifier/cleaned_train_data_{detector_name}_{dataset_name}.csv"), row.names = FALSE)
  }
}
