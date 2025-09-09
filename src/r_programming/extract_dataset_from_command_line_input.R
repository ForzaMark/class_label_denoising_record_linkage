extract_dataset_from_command_line_input <- function(args) {
  dataset_name <- if (length(args) >= 1) args[1] else NULL
  
  if (is.null(dataset_name) || !(dataset_name %in% c("dexter", "wdc_almser", "music"))) {
    stop("Invalid or missing dataset. Must be one of: 'dexter', 'wdc_almser', or 'music'")
  }

  general_dataset <- dataset_name
  dataset <- if (general_dataset %in% c("music", "wdc_almser")) {
    paste0(general_dataset, "_most_values")
  } else {
    general_dataset
  }

  return(list(general_dataset = general_dataset, dataset = dataset))
}
