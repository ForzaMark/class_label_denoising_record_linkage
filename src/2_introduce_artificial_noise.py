import pandas as pd
from util.corrupt_data import corrupt_data
from util.get_out_of_sample_predicted_probabilities_from_teacher_model import get_out_of_sample_predicted_probabilities_from_teacher_model
from util.create_bins import create_bins
from util.save_to_csv import save_to_csv
from util.compute_mislabeling_probabilities import compute_mislabeling_probabilities
from constants.teacher_models import TEACHER_MODELS
import sys
from util.extract_dataset_from_command_line_input import extract_dataset_from_command_line_input

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    general_dataset, dataset = extract_dataset_from_command_line_input(sys.argv[1])

    for (teacher_name, teacher_model) in TEACHER_MODELS:
        data = pd.read_csv(f"../datasets/{general_dataset}/preprocessed_{dataset}.csv")

        input_data = data.copy()
        binning_method = 'clustering_efficient'
        feature_data = input_data.drop(['label'], axis=1)
        bins = create_bins(feature_data, binning_method, dataset)

        data = get_out_of_sample_predicted_probabilities_from_teacher_model(data, clf=teacher_model, stratified_training=True, scale_feature_data=teacher_name == 'svm')
        data["bin"] = bins
        
        data = compute_mislabeling_probabilities(data)

        data = data.drop(["predicted_label", "bin"], axis=1)

        corrupted_data = corrupt_data(data)
        corrupted_data = corrupted_data.drop("mislabeling_probability", axis=1)

        print(f'Introduced noise rate {((len(corrupted_data[corrupted_data["label"] != corrupted_data["noisy_label"]]) / len(corrupted_data)) * 100):.3f} %')

        match_prior = len(corrupted_data[corrupted_data["label"] == 1]) / len(corrupted_data)
        non_match_prior = len(corrupted_data[corrupted_data["label"] == 0]) / len(corrupted_data)

        print(f'p(y_tilde = 0 | y*= 1) {(((len(corrupted_data[(corrupted_data["label"] == 1) & (corrupted_data["noisy_label"] == 0)]) / len(corrupted_data)) / match_prior) * 100):.2f} %')
        print(f'p(y_tilde = 1 | y*= 0) {(((len(corrupted_data[(corrupted_data["label"] == 0) & (corrupted_data["noisy_label"] == 1)]) / len(corrupted_data)) / non_match_prior) * 100):.2f} %')

        print(f'Imbalance: percentage matches = {((len(corrupted_data[(corrupted_data["label"] == 0)]) / len(corrupted_data)) * 100):.2f} %')

        save_to_csv(corrupted_data, f"../datasets/{general_dataset}/{dataset}_{teacher_name}_corrupted.csv")