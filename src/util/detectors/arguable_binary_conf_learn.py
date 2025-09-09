import pandas as pd

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from cleanlab import Datalab

from sklearn.cluster import KMeans

def extract_original_label(input):
    return input.split("_")[1]


def partition_feature_space_kmeans(k, feature_data):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(feature_data)

    return kmeans.labels_

class Arguable_Binary_Conf_Learn:
    def __init__(self, k=5):
        self.k = k


    def detect(self, feature_data, noisy_labels):
        bins = partition_feature_space_kmeans(self.k, feature_data)

        base_data = feature_data.copy()
        base_data["label"] = noisy_labels
        base_data["bin"] = bins

        base_data["multi_class"] = base_data.apply(lambda row: f'{int(row["bin"])}_{int(row["label"])}', axis=1)
        base_data = base_data[base_data['multi_class'].map(base_data['multi_class'].value_counts()) > 10]


        X_raw = base_data.drop(["label", "bin", "multi_class"], axis=1)
        labels = base_data["multi_class"]

        scaler = StandardScaler()
        X_processed = X_raw.copy()
        X_processed = scaler.fit_transform(X_raw)

        clf = HistGradientBoostingClassifier()
        pred_probs = cross_val_predict(
            clf,
            X_processed,
            labels,
            method="predict_proba",
        )

        KNN = NearestNeighbors(metric='euclidean')
        KNN.fit(X_processed)

        knn_graph = KNN.kneighbors_graph(mode="distance")  

        data = {"X": X_processed, "y": labels}

        lab = Datalab(data, label_name="y", verbosity=False)
        lab.find_issues(pred_probs=pred_probs, knn_graph=knn_graph)

        issues = lab.get_issues('label')
        combined = pd.concat([issues, feature_data], axis=1)
        label_issue_instances = combined[combined['is_label_issue']==True]

        label_issue_instances["actually_mislabeled"] = label_issue_instances.apply(lambda row: extract_original_label(row["given_label"]) != extract_original_label(row["predicted_label"]), axis=1)

        label_issue_instances = label_issue_instances[label_issue_instances["actually_mislabeled"]]

        return label_issue_instances.index.to_numpy()
