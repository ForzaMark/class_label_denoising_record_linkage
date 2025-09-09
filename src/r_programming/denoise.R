#install.packages("./noise_filters_r/NoiseFiltersR_0.1.0.tar.gz", repos = NULL, type = "source")

setwd("~/label_denoising_record_linkage/src/r_programming")

library("NoiseFiltersR")
library(dplyr)
library(jsonlite)
library(glue)

source("calculate_own_evaluation_measures.R")

exclude_label_from_features <- function(noisy_data) {
  return (names(noisy_data)[names(noisy_data) != "label"])
}

denoise <- function(datasets, repetition_identifier, save_csv = TRUE) {
  detectors <- list(
    "cvcf",
    "harf"
  )
  
  for (detector_name in detectors) {
    results <- list()
    for (dataset_triple in datasets) {
      dataset <- dataset_triple[1]
      dataset_name <- dataset_triple[2]
      file <- dataset_triple[3]
      
      general_dataset <- ifelse(grepl("dexter", dataset), "dexter",
          ifelse(grepl("music", dataset), "music", "wdc_almser"))

      detector = ifelse(detector_name == 'harf', HARF, CVCF)
      
      print(detector_name)
      print(dataset_name)
      
      noisy_data <- read.csv(file)
      features = exclude_label_from_features(noisy_data)
      
      noise_rate = sum(noisy_data$label != noisy_data$noisy_label) / nrow(noisy_data)
      data <- noisy_data %>% select(all_of(features))
      data$noisy_label <- as.factor(data$noisy_label)
      
      start_time <- Sys.time()
      out <- detector(noisy_label~., data = data)
      end_time <- Sys.time()
      
      noisy_indices <- which(noisy_data$noisy_label != noisy_data$label)
      y_true <- ifelse(1:(nrow(noisy_data)) %in% noisy_indices, 1, 0)
      y_pred <- ifelse(1:(nrow(noisy_data)) %in% out$remIdx, 1, 0)
      
      TP <- sum(y_true == 1 & y_pred == 1)
      TN <- sum(y_true == 0 & y_pred == 0)
      FP <- sum(y_true == 0 & y_pred == 1)
      FN <- sum(y_true == 1 & y_pred == 0)
      
      
      accuracy <- (TP + TN) / (TP + TN + FP + FN)
      precision <- TP / (TP + FP)
      precision_over_noise <- ifelse(noise_rate != 0, precision - noise_rate, "unable to compute noise over precision")
      recall <- TP / (TP + FN)
      f1 <- 2 * precision * recall / (precision + recall)
      beta <- 0.5
      f05 <- (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
      
      if ((TP + FP) == 0) precision <- 0
      if ((TP + FN) == 0) recall <- 0
      if ((precision + recall) == 0) f1 <- 0
      if ((precision + recall) == 0) f05 <- 0
      
      time_diff <- end_time - start_time
      
      proportion_minority_in_fp <- get_own_measures(out$remIdx, noisy_data)
      
      result <- list(
        dataset = dataset_name,
        detector = detector_name,
        noise_rate = noise_rate,
        time = as.character(time_diff),
        cleaning = list(
          accuracy = accuracy,
          precision = precision,
          recall = recall,
          f1 = f1,
          f05 = f05,
          pon = precision_over_noise,
          fp_proportion = proportion_minority_in_fp
        )
      )
      
      results[[length(results) + 1]] <- result
      
      if (length(out$remIdx) > 0) {
        cleaned_data <- noisy_data[-out$remIdx, ]
      } else {
        cleaned_data = noisy_data
      }
      
      if (save_csv) {
        write.csv(cleaned_data, glue("../../datasets/{dataset}/{dataset_name}_{detector_name}_cleaned.csv"), row.names = FALSE)
      }
      
      write_json(results, path = glue("../../results/evaluation_{general_dataset}/{detector_name}_{repetition_identifier}.json"), pretty = TRUE, auto_unbox = TRUE)
    }
  }
}




