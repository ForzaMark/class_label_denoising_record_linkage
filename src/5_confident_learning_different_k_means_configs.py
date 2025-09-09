from sklearn.metrics import fbeta_score
import pandas as pd
import numpy as np
from util.evaluation_metrics import get_evaluation_metrics
from util.detectors.arguable_binary_conf_learn import Arguable_Binary_Conf_Learn, partition_feature_space_kmeans
from util.detectors.conf_learn_detector import Conf_Learn_Detector
import json
from constants.teacher_models import TEACHER_MODELS
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

def create_different_clustering_configurations(feature_data, noisy_labels):
    clusterings = []

    for k in [2, 3, 4, 5, 6, 8, 10, 20, 40]:
        bins = partition_feature_space_kmeans(k, feature_data)
        
        corrupted_data_multiclass = feature_data.copy()
        corrupted_data_multiclass["label"] = noisy_labels
        corrupted_data_multiclass["bin"] = bins
        corrupted_data_multiclass["noisy_label"] = corrupted_data_multiclass.apply(lambda row: f'{int(row["bin"])}_{int(row["label"])}', axis=1)

        corrupted_data_multiclass = corrupted_data_multiclass.drop(["bin", "label"], axis=1)

        clusterings.append((k, corrupted_data_multiclass))

    return clusterings

def filter_clusterings_with_less_200_instances_in_one_class(clusterings):
    reasonable_clusterings = []

    for (k, clustering) in clusterings:
        per_class_sizes = clustering.groupby("noisy_label").size()
        has_classes_with_less_100_instances = len([size for size in per_class_sizes if size < 200]) > 0

        if not has_classes_with_less_100_instances:
            reasonable_clusterings.append((k, clustering))

    return reasonable_clusterings

def compute_denoising_performance(detector, corrupted_data):
    feature_data = corrupted_data.drop(["label", "noisy_label"], axis=1)
    noisy_labels = corrupted_data["noisy_label"] 

    issue_indices = detector.detect(feature_data, noisy_labels)
    
    y_pred = [1 if i in issue_indices else 0 for i in np.arange(0, len(corrupted_data), 1)]
    y_true = corrupted_data.apply(lambda row: 0 if row["label"] == row["noisy_label"] else 1, axis=1).to_numpy()

    noise_rate = len(corrupted_data[corrupted_data["label"] != corrupted_data["noisy_label"]]) / len(corrupted_data)

    _, precision, recall, _, _, cm, _ = get_evaluation_metrics(y_true, y_pred, noise_rate)
    f2 = fbeta_score(y_true, y_pred, beta=2)

    return recall, f2

def get_label_order(clf, X_processed, labels):
    clf.fit(X_processed, labels)

    return clf.classes_

def compute_thresholds(clustering):
    thresholds = []

    data = clustering.copy()
    feature_data = data.drop(["noisy_label"], axis=1)
    noisy_labels = data["noisy_label"] 

    X_raw = feature_data
    labels = noisy_labels

    scaler = StandardScaler()
    X_processed = X_raw.copy()
    X_processed = scaler.fit_transform(feature_data)

    clf = HistGradientBoostingClassifier()

    pred_probs = cross_val_predict(
        clf,
        X_processed,
        labels,
        method="predict_proba",
    )

    class_columns = [f'predicted_probability_class{label}' for label in get_label_order(clf, X_processed, labels)]

    data[class_columns] = pred_probs

    for noisy_label in data["noisy_label"].unique():
        class_instances = data[data["noisy_label"] == noisy_label]
        threshold = (1 / len(class_instances)) * sum(class_instances[f"predicted_probability_class{noisy_label}"])

        thresholds.append(threshold)

    return thresholds

def filter_clusterings_which_do_not_decrease_threshold_mean(clusterings, base_data):
    base_data = base_data.drop(["label"], axis=1)
    base_threshold_mean = np.mean(compute_thresholds(base_data))

    reasonable_clusterings = []

    for (k, clustering) in clusterings:
        clustering_threshold_mean = np.mean(compute_thresholds(clustering))

        if base_threshold_mean - clustering_threshold_mean > 0.05:
            reasonable_clusterings.append((k, clustering))

    return reasonable_clusterings

DATASETS = [
    ('music', 'music_most_values'),
    ("wdc_almser", "wdc_almser_most_values"),
    ("dexter", "dexter")
]

if __name__ == '__main__':
    results = []

    for (model_name, model) in TEACHER_MODELS:
        for (general_dataset, dataset) in DATASETS:
            corrupted_data = pd.read_csv(f'../datasets/{general_dataset}/{dataset}_{model_name}_corrupted.csv')

            feature_data = corrupted_data.drop(["noisy_label", "label"], axis=1)
            noisy_labels = corrupted_data["noisy_label"]

            clusterings = create_different_clustering_configurations(feature_data, noisy_labels)

            reasonable_clusterings = filter_clusterings_with_less_200_instances_in_one_class(clusterings)

            reasonable_clusterings = filter_clusterings_which_do_not_decrease_threshold_mean(reasonable_clusterings, corrupted_data)

            dataset_results = []
        
            base_recall, base_f2 = compute_denoising_performance(Conf_Learn_Detector(), corrupted_data)

            dataset_results.append({
                'dataset': f'{dataset}_{model_name}',
                'k': 'baseline',
                'recall': base_recall,
                'f2': base_f2
            })

            for (k, clustering) in reasonable_clusterings:
                recall, f2 = compute_denoising_performance(Arguable_Binary_Conf_Learn(k), corrupted_data)

                dataset_results.append({
                    'dataset': f'{dataset}_{model_name}',
                    'k': k,
                    'recall': recall,
                    'f2': f2
                })

            results.append(dataset_results)

            with open(f'../results/evaluation_conf_learn_adjustment/results.json', 'w') as f:
                json.dump(results, f)
