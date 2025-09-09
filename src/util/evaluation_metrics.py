
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import numpy as np

def get_evaluation_metrics(y_true, y_pred, noise_rate):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f05 = fbeta_score(y_true, y_pred, beta=0.5)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    precision_over_noise = precision - noise_rate if noise_rate else 'unable to compute noise over precision'

    return acc, precision, recall, f1, f05, cm, precision_over_noise

def fp_proportion(y_pred, y_true, corrupted_data):
    fp_indices = [index for index, value in enumerate(y_pred) if value == 1 and y_true[index] == 0]

    fps = corrupted_data.iloc[fp_indices]

    matches_in_false_positives = len(fps[fps["label"] == 1])
    non_matches_in_false_positives = len(fps[fps["label"] == 0])

    p_fp_m = matches_in_false_positives / len(corrupted_data[(corrupted_data["label"] == 1) & (corrupted_data["noisy_label"] == 1)])
    p_fp_nm = non_matches_in_false_positives / len(corrupted_data[(corrupted_data["label"] == 0) & (corrupted_data["noisy_label"] == 0)])

    return p_fp_m / np.clip(p_fp_nm, a_min=0.0000001, a_max=None)