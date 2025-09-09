from scipy.spatial.distance import pdist
import numpy as np

def get_single_max_distance_of_two_elements_in_single_cluster(evaluation_data):
    max_value = 0
    for bin, df in evaluation_data.groupby('bin'):
        if len(df) != 1:
            features = df.drop(["bin"], axis=1)
            distances = pdist(features, metric='euclidean')

            max_distance = distances.max()

            if max_distance > max_value:
                max_value = max_distance

    return max_value

def get_average_max_distance_between_two_elements_in_cluster(evaluation_data):
    max_values = []
    for bin, df in evaluation_data.groupby('bin'):
        if len(df) != 1:
            features = df.drop(["label", "bin"], axis=1)
            distances = pdist(features, metric='euclidean')

            max_distance = distances.max()

            max_values.append(max_distance)

    return np.array(max_values).mean()

def get_max_distance_of_element_to_its_prototype(evaluation_data, bin_prototypes):
    max_dist = 0

    bin_assignment = evaluation_data["bin"].to_numpy()

    data = evaluation_data.drop("bin", axis=1)

    for index, feature_vector in enumerate(data.to_numpy()):
        prototype = bin_prototypes[bin_assignment[index]]

        distance = np.linalg.norm(prototype - feature_vector)

        if distance > max_dist:
            max_dist = distance

    return max_dist
        