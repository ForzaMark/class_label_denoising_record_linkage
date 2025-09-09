library(mice)
library(glue)

setwd("~/label_denoising_record_linkage/src/r_programming")


datasets <- list(
  "music",
  "wdc_almser"
)

for (dataset in datasets) {
  print(dataset)
  data_to_impute <- read.csv(glue("../../datasets/{dataset}/preprocessed_{dataset}_full.csv"))
  
  labels <- data_to_impute$label
  data_to_impute$label <- NULL
  
  imputed <- mice(data_to_impute, method='pmm')
  
  music_imputed <- complete(imputed, action = 1)
  music_imputed$label <- labels
  
  write.csv(music_imputed, glue("../../datasets/{dataset}/preprocessed_{dataset}_imputed.csv"), row.names = FALSE)
}


