import sys
from util.detectors.conf_learn_detector import Conf_Learn_Detector
from util.detectors.fkdn_detector import FKDN_Detector
from util.detectors.arguable_binary_conf_learn import Arguable_Binary_Conf_Learn
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob
import json
from util.get_information_from_file_path import get_dataset_from_file_path
from constants.teacher_models import TEACHER_MODELS

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from util.extract_dataset_from_command_line_input import extract_dataset_from_command_line_input

DETECTORS = [
    ('conf_learn', Conf_Learn_Detector), 
    ('arguable_binary_conf_learn', Arguable_Binary_Conf_Learn),
    ('fkdn', FKDN_Detector)
]

def train_evaluate_classifier(x_train, y_train, x_test, y_test):

    assert 'label' not in x_train.columns 
    assert 'noisy_label' not in x_train.columns 

    assert 'label' not in x_test.columns 
    assert 'noisy_label' not in x_test.columns 

    base_estimator = DecisionTreeClassifier(max_depth=1)

    clf = AdaBoostClassifier(estimator=base_estimator)

    clf.fit(x_train, y_train)

    y_true = y_test
    y_pred = clf.predict(x_test)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1

if __name__ == '__main__':
    mode = sys.argv[1]
    print(mode)

    general_dataset, specific_dataset = extract_dataset_from_command_line_input(sys.argv[2])

    if mode == 'train_test_split':
        for (teacher_name, teacher_model) in TEACHER_MODELS:
                corrupted_data = pd.read_csv(f'../datasets/{general_dataset}/{specific_dataset}_{teacher_name}_corrupted.csv')

                train_df, test_df = train_test_split(corrupted_data, test_size=0.2)

                clean_train_data = train_df.drop(["noisy_label"], axis=1)
                corrupted_train_data = train_df.drop("label", axis=1).rename(columns={'noisy_label': 'label'})
                clean_test_data = test_df.drop(["noisy_label"], axis=1)

                clean_train_data.to_csv(f"../datasets/evaluation_scenarios/experiment_down_stream_classifier/{specific_dataset}_{teacher_name}_clean_train_data.csv", index=False)
                corrupted_train_data.to_csv(f"../datasets/evaluation_scenarios/experiment_down_stream_classifier/{specific_dataset}_{teacher_name}_corrupted_train_data.csv", index=False)
                clean_test_data.to_csv(f"../datasets/evaluation_scenarios/experiment_down_stream_classifier/{specific_dataset}_{teacher_name}_clean_test_data.csv", index=False)

    if mode == 'cleaning':
        corrupted_train_data_files = glob(f'../datasets/evaluation_scenarios/experiment_down_stream_classifier/{specific_dataset}*_corrupted_train_data.csv')

        for file in corrupted_train_data_files:
            dataset_name = get_dataset_from_file_path(file)
            print(dataset_name)

            for detector_name, detector in DETECTORS:
                print(detector_name)

                teacher_name = 'rf' if 'rf' in file else 'svm' if 'svm'  in file else 'tree'

                corrupted_data = pd.read_csv(file)
                corrupted_feature_data = corrupted_data.drop("label", axis=1)

                detector_instance = detector()
                if detector_name == 'binary_conf_learn':
                    issue_indices = detector_instance.detect(corrupted_feature_data, corrupted_data["label"].to_numpy(), dataset_name)
                else:
                    issue_indices = detector_instance.detect(corrupted_feature_data, corrupted_data["label"].to_numpy())

                cleaned_data = corrupted_data.drop(index=issue_indices)
                cleaned_data.to_csv(f'../datasets/evaluation_scenarios/experiment_down_stream_classifier/cleaned_train_data_{detector_name}_{specific_dataset}_{teacher_name}.csv', index=False)

    if mode == 'evaluation':
        results = []

        for (teacher_name, _) in TEACHER_MODELS: 
                clean_train_data = pd.read_csv(f"../datasets/evaluation_scenarios/experiment_down_stream_classifier/{specific_dataset}_{teacher_name}_clean_train_data.csv")
                corrupted_train_data = pd.read_csv(f"../datasets/evaluation_scenarios/experiment_down_stream_classifier/{specific_dataset}_{teacher_name}_corrupted_train_data.csv")
                clean_test_data = pd.read_csv(f"../datasets/evaluation_scenarios/experiment_down_stream_classifier/{specific_dataset}_{teacher_name}_clean_test_data.csv")

                cleaned_conf_learn_data = pd.read_csv(f'../datasets/evaluation_scenarios/experiment_down_stream_classifier/cleaned_train_data_conf_learn_{specific_dataset}_{teacher_name}.csv')
                cleaned_binary_conf_learn_data = pd.read_csv(f'../datasets/evaluation_scenarios/experiment_down_stream_classifier/cleaned_train_data_arguable_binary_conf_learn_{specific_dataset}_{teacher_name}.csv')
                cleaned_cvcf_data = pd.read_csv(f'../datasets/evaluation_scenarios/experiment_down_stream_classifier/cleaned_train_data_cvcf_{specific_dataset}_{teacher_name}.csv')
                cleaned_fkdn_data = pd.read_csv(f'../datasets/evaluation_scenarios/experiment_down_stream_classifier/cleaned_train_data_fkdn_{specific_dataset}_{teacher_name}.csv')
                cleaned_harf_data = pd.read_csv(f'../datasets/evaluation_scenarios/experiment_down_stream_classifier/cleaned_train_data_harf_{specific_dataset}_{teacher_name}.csv')

                clean_precision, clean_recall, clean_f1 = train_evaluate_classifier(clean_train_data.drop("label", axis=1), 
                                                                clean_train_data["label"], 
                                                                clean_test_data.drop("label", axis=1), 
                                                                clean_test_data["label"])
                
                corrupted_precision, corrupted_recall, corrupted_f1 = train_evaluate_classifier(corrupted_train_data.drop("label", axis=1), 
                                                                corrupted_train_data["label"], 
                                                                clean_test_data.drop("label", axis=1), 
                                                                clean_test_data["label"])
                
                cleaned_conf_learn_precision, cleaned_conf_learn_recall, cleaned_conf_learn_f1 = train_evaluate_classifier(cleaned_conf_learn_data.drop("label", axis=1), 
                                                                cleaned_conf_learn_data["label"], 
                                                                clean_test_data.drop("label", axis=1), 
                                                                clean_test_data["label"])
                
                cleaned_binary_conf_learn_precision, cleaned_binary_conf_learn_recall, cleaned_binary_conf_learn_f1 = train_evaluate_classifier(cleaned_binary_conf_learn_data.drop("label", axis=1), 
                                                                cleaned_binary_conf_learn_data["label"], 
                                                                clean_test_data.drop("label", axis=1), 
                                                                clean_test_data["label"])
                
                cleaned_cvcf_data = cleaned_cvcf_data.rename(columns={
                    'Part.Number_lev': 'Part Number_lev',
                    'Part.Number_jaccard': 'Part Number_jaccard', 
                    'Part.Number_relaxed_jaccard': 'Part Number_relaxed_jaccard',
                    'Part.Number_overlap': 'Part Number_overlap', 
                    'Part.Number_containment': 'Part Number_containment',
                    'Part.Number_token_jaccard': 'Part Number_token_jaccard'})

                
                cleaned_cvcf_precision, cleaned_cvcf_recall, cleaned_cvcf_f1 = train_evaluate_classifier(cleaned_cvcf_data.drop("label", axis=1), 
                                                                cleaned_cvcf_data["label"], 
                                                                clean_test_data.drop("label", axis=1), 
                                                                clean_test_data["label"])

                cleaned_fkdn_precision, cleaned_fkdn_recall, cleaned_fkdn_f1 = train_evaluate_classifier(cleaned_fkdn_data.drop("label", axis=1), 
                                                                cleaned_fkdn_data["label"], 
                                                                clean_test_data.drop("label", axis=1), 
                                                                clean_test_data["label"])
                
                cleaned_harf_data = cleaned_harf_data.rename(columns={
                    'Part.Number_lev': 'Part Number_lev',
                    'Part.Number_jaccard': 'Part Number_jaccard', 
                    'Part.Number_relaxed_jaccard': 'Part Number_relaxed_jaccard',
                    'Part.Number_overlap': 'Part Number_overlap', 
                    'Part.Number_containment': 'Part Number_containment',
                    'Part.Number_token_jaccard': 'Part Number_token_jaccard'})
                
                cleaned_harf_precision, cleaned_harf_recall, cleaned_harf_f1 = train_evaluate_classifier(cleaned_harf_data.drop("label", axis=1), 
                                                                cleaned_harf_data["label"], 
                                                                clean_test_data.drop("label", axis=1), 
                                                                clean_test_data["label"])

                results.append({
                    'dataset': f"{specific_dataset}_{teacher_name}",
                    
                    "clean_precision": clean_precision,
                    "clean_recall": clean_recall,
                    "clean_f1": clean_f1,
                    
                    "corrupted_precision": corrupted_precision,
                    "corrupted_recall": corrupted_recall,
                    "corrupted_f1": corrupted_f1,
                    
                    "cleaned_conf_learn_precision": cleaned_conf_learn_precision,
                    "cleaned_conf_learn_recall": cleaned_conf_learn_recall,
                    "cleaned_conf_learn_f1": cleaned_conf_learn_f1,

                    "cleaned_binary_conf_learn_precision": cleaned_binary_conf_learn_precision,
                    "cleaned_binary_conf_learn_recall": cleaned_binary_conf_learn_recall,
                    "cleaned_binary_conf_learn_f1": cleaned_binary_conf_learn_f1,
                    
                    "cleaned_cvcf_precision": cleaned_cvcf_precision,
                    "cleaned_cvcf_recall": cleaned_cvcf_recall,
                    "cleaned_cvcf_f1": cleaned_cvcf_f1,
                    
                    "cleaned_fkdn_precision": cleaned_fkdn_precision,
                    "cleaned_fkdn_recall": cleaned_fkdn_recall,
                    "cleaned_fkdn_f1": cleaned_fkdn_f1,
                    
                    "cleaned_harf_precision": cleaned_harf_precision,
                    "cleaned_harf_recall": cleaned_harf_recall,
                    "cleaned_harf_f1": cleaned_harf_f1
                })


                with open(f'../results/evaluation_{general_dataset}/experiment_down_stream_classifier.json', 'w') as f:
                    json.dump(results, f)


