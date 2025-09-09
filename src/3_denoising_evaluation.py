from util.detectors.conf_learn_detector import Conf_Learn_Detector
from util.detectors.arguable_binary_conf_learn import Arguable_Binary_Conf_Learn
from util.detectors.fkdn_detector import FKDN_Detector
import pandas as pd
from util.evaluation_metrics import get_evaluation_metrics, fp_proportion
import time
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import sys
from util.extract_dataset_from_command_line_input import extract_dataset_from_command_line_input

DETECTORS = [
    ('conf_learn', Conf_Learn_Detector),
    ('arguable_conf_learn', Arguable_Binary_Conf_Learn),
    ('fkdn', FKDN_Detector),
]

TEACHER_MODELS = [
     ('rf', RandomForestClassifier()),
     ('svm', LinearSVC()),
     ('tree', DecisionTreeClassifier())
]

if __name__ == "__main__":
    results = []

    general_dataset, dataset = extract_dataset_from_command_line_input(sys.argv[1])
    
    for (model_name, model) in TEACHER_MODELS:
        for detector_name, detector in DETECTORS: 
            print(dataset, detector_name, model_name)

            corrupted_data = pd.read_csv(f'../datasets/{general_dataset}/{dataset}_{model_name}_corrupted.csv')
            corrupted_feature_data = corrupted_data.drop(["label", "noisy_label"], axis=1)

            start_time = time.time()

            detector_instance = detector()
            if detector_name == 'binary_conf_learn':
                issue_indices = detector_instance.detect(corrupted_feature_data, corrupted_data["noisy_label"].to_numpy(), dataset)
            else:
                issue_indices = detector_instance.detect(corrupted_feature_data, corrupted_data["noisy_label"].to_numpy())

            end_time = time.time()

            y_pred = [1 if i in issue_indices else 0 for i in np.arange(0, len(corrupted_data), 1)]
            y_true = corrupted_data.apply(lambda row: 0 if row["label"] == row["noisy_label"] else 1, axis=1).to_numpy()

            noise_rate = len(corrupted_data[corrupted_data["label"] != corrupted_data["noisy_label"]]) / len(corrupted_data)

            acc, precision, recall, f1, f05, cm, precision_over_noise = get_evaluation_metrics(y_true, y_pred, noise_rate)
            proportion = fp_proportion(y_pred, y_true, corrupted_data)

            time_in_seconds = end_time - start_time

            result = {
                'dataset': f"{dataset}_{model_name}_corrupted",
                'model': model_name,
                'detector': detector_name,
                'noise_rate': noise_rate,
                'time': time_in_seconds,
                'cleaning': {
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'f05': f05,
                    'pon': precision_over_noise,
                    'fp_proportion': proportion
                }
            }

            results.append(result)

            with open(f'../results/evaluation_{general_dataset}/conf_learn_fkdn_evaluation.json', 'w') as f:
                json.dump(results, f)
