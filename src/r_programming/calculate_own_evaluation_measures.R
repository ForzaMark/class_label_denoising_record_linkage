get_own_measures <- function(removed_indices, noisy_data) {
  detected_mislabeled <- noisy_data[removed_indices, ]
  false_positive_instances <- detected_mislabeled[detected_mislabeled$label == detected_mislabeled$noisy_label, ]
  
  matches_in_false_positives <- nrow(false_positive_instances[false_positive_instances$label == 1,])
  non_matches_in_false_positives <- nrow(false_positive_instances[false_positive_instances$label == 0,])
  
  matches_in_noisy_data <- nrow(noisy_data[noisy_data$label == 1 & noisy_data$noisy_label == 1,])
  non_matches_in_noisy_data <- nrow(noisy_data[noisy_data$label == 0 & noisy_data$noisy_label == 0,])
  
  p_fp_m = matches_in_false_positives / matches_in_noisy_data
  p_fp_nm = non_matches_in_false_positives / non_matches_in_noisy_data
  
  return(p_fp_m / p_fp_nm)
}
