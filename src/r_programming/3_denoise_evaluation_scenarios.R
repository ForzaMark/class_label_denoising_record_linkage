setwd("~/label_denoising_record_linkage/src/r_programming")

source("denoise.R")
source("extract_dataset_from_command_line_input.R")

args <- commandArgs(trailingOnly = TRUE)
result <- extract_dataset_from_command_line_input(args)

general_dataset <- result$general_dataset
dataset <- result$dataset

datasets <- list(
  c(dataset, glue("{dataset}_rf_corrupted"), glue("../../datasets/{general_dataset}/{dataset}_rf_corrupted.csv")),
  c(dataset, glue("{dataset}_svm_corrupted"), glue("../../datasets/{general_dataset}/{dataset}_svm_corrupted.csv")),
  c(dataset, glue("{dataset}_tree_corrupted"), glue( "../../datasets/{general_dataset}/{dataset}_tree_corrupted.csv"))
)

repetition_identifier <- format(Sys.time(), "%Y-%m-%d-%H-%M-%S")

denoise(datasets, repetition_identifier, save_csv = FALSE)