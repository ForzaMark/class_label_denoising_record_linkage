import pandas as pd

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from cleanlab import Datalab


class Conf_Learn_Detector:
    def detect(self, feature_data, noisy_labels):
        X_raw = feature_data
        labels = noisy_labels

        scaler = StandardScaler()
        X_processed = X_raw.copy()
        X_processed = scaler.fit_transform(feature_data)

        clf = HistGradientBoostingClassifier()
        num_crossval_folds = 5 if len(feature_data) > 5 else 2
        pred_probs = cross_val_predict(
            clf,
            X_processed,
            labels,
            cv=num_crossval_folds,
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

        return label_issue_instances.index.to_numpy()

