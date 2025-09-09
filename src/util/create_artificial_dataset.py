import numpy as np
import pandas as pd
import random
from util.get_out_of_sample_predicted_probabilities_from_teacher_model import get_out_of_sample_predicted_probabilities_from_teacher_model
from util.create_bins import create_bins
from util.compute_mislabeling_probabilities import compute_mislabeling_probabilities
from util.corrupt_data import corrupt_data

# Creation of this dataset inspired by Menon et al: Learning from Binary Labels with Instance-Dependent Corruption
def create_artificial_dataset(n_samples_class1 = 2500, n_samples_class2 = 2500):
    mean1 = [1, 1]
    mean2 = [-1, -1]
    cov = np.identity(2) 

    samples1 = np.random.multivariate_normal(mean1, cov, n_samples_class1)
    samples2 = np.random.multivariate_normal(mean2, cov, n_samples_class2)

    labels1 = np.zeros(n_samples_class1, dtype=int)
    labels2 = np.ones(n_samples_class2, dtype=int)

    all_samples = np.vstack((samples1, samples2))
    all_labels = np.hstack((labels1, labels2))

    df = pd.DataFrame(all_samples, columns=['x1', 'x2'])
    df['label'] = all_labels

    return df

def corrupt_sln(df, noise_rate): 
    sln_corrupted_data = df.copy()
    sln_corrupted_data["noisy_label"] = sln_corrupted_data.apply(lambda row: row["label"] if random.random() > noise_rate else int(abs(row["label"] - 1)), axis=1)
    return sln_corrupted_data

def corrupt_ccn_internal(row, mislabeling_class0, mislabeling_class1):
    label = row["label"]

    if label == 0:
        if random.random() > mislabeling_class0:
            return label
        else:
            return int(abs(row["label"] - 1))
    else:
        if random.random() > mislabeling_class1:
            return label
        else:
            return int(abs(row["label"] - 1))


def corrupt_ccn(df, alpha, beta):
    ccn_corrupted_data = df.copy()
    mislabeling_class0 = alpha
    mislabeling_class1 = beta
    ccn_corrupted_data["noisy_label"] = ccn_corrupted_data.apply(lambda row: corrupt_ccn_internal(row, mislabeling_class0, mislabeling_class1), axis=1)
    return ccn_corrupted_data

def corrupt_iln(df, teacher_model, dataset):
    input_data = df.copy()

    binning_method = 'clustering_efficient'
    feature_data = input_data.drop(['label'], axis=1)
    bins = create_bins(feature_data, binning_method, dataset)

    data = get_out_of_sample_predicted_probabilities_from_teacher_model(input_data, clf=teacher_model, stratified_training=True)
    data["bin"] = bins
    
    data = compute_mislabeling_probabilities(data)
    data = data.drop(["predicted_label", "bin"], axis=1)
    iln_corrupted_data = corrupt_data(data)

    return iln_corrupted_data