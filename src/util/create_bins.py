from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
import numpy as np
import pandas as pd
from util.cluster_max_distances import get_max_distance_of_element_to_its_prototype, get_single_max_distance_of_two_elements_in_single_cluster
import hdbscan

PARTITIONING_MEASURE_CUTOFF_LOOKUP = {
    'artificial': 1,
    'dexter': 13,
    'music_most_values': 3,
    'wdc_almser_most_values': 30,
    'music_adjusted_conf_learn': 1
}

def create_bins(feature_data, binning_method, dataset):

    assert len([c for c in feature_data.columns if 'label' in c]) == 0

    if binning_method == 'clustering':
        return create_bins_clustering(feature_data, linkage_method='ward', partitioning_measure_cutoff=15)
    if binning_method == 'clustering_efficient':
        partitioning_measure_cutoff = PARTITIONING_MEASURE_CUTOFF_LOOKUP[dataset]
        return create_bins_clustering_efficient(feature_data, linkage_method='ward', partitioning_measure_cutoff=partitioning_measure_cutoff)
    if binning_method == 'pca':
        return create_bins_pca_projection(feature_data, number_pca_components=2, number_bins_per_reduced_dimension=10)
    if binning_method == 'geometric':
        return create_bins_geometric_sampling(feature_data, number_of_bins_per_dimension=2)

def calculate_prototypes(sampled_feature_data):
    prototypes = []
    for index, bin in sampled_feature_data.groupby("bin"):
        bin = bin.drop("bin", axis=1)
        prototype_vector = []

        for column in bin.columns:
            prototype_vector.append(bin[column].mean())


        prototypes.append(prototype_vector)

    return prototypes

def calculate_all_bins(feature_data, bin_prototypes, max_distance):
    result = []
    for feature_vector in feature_data.to_numpy():
        distances = np.linalg.norm(bin_prototypes - feature_vector, axis=1)
        closest_index = np.argmin(distances)

        value = distances[closest_index]

        if value > max_distance:
            result.append(len(bin_prototypes) + 1)
            bin_prototypes.append(feature_vector)
        else:
            result.append(closest_index)

    return result


def create_bins_clustering_efficient(feature_data, linkage_method, partitioning_measure_cutoff):
    max_feasible_len = 6000

    if len(feature_data) > max_feasible_len:
        sampled_feature_data = feature_data.sample(n=max_feasible_len)
        sampled_feature_data["bin"] = create_bins_clustering(sampled_feature_data, linkage_method, partitioning_measure_cutoff)

        bin_prototypes = calculate_prototypes(sampled_feature_data)
        max_distance = get_single_max_distance_of_two_elements_in_single_cluster(sampled_feature_data)

        return calculate_all_bins(feature_data, bin_prototypes, max_distance)

    else:
        return create_bins_clustering(feature_data, linkage_method, partitioning_measure_cutoff)
    

def create_bins_geometric_sampling(feature_data, number_of_bins_per_dimension):
    bin_indices = np.floor(feature_data * number_of_bins_per_dimension).astype(int)
    bin_indices = np.clip(bin_indices, 0, number_of_bins_per_dimension - 1)

    bin_assignments = []

    for i in range(len(bin_indices)):
        bin_assignments.append(''.join([str(value) for value in bin_indices.iloc[i].values]))

    return bin_assignments

def calculate_partitioning_measure(current_partition):
    result = 0

    for partition in current_partition:
        result = result + (1 / len(partition))

    return result
    

def create_bins_clustering(feature_data, 
                           linkage_method, 
                           partitioning_measure_cutoff):
    X = feature_data.to_numpy()
    Z = linkage(X, method=linkage_method)
    n = len(X)

    clusters = {i: [i] for i in range(n)}
    best_partition = None

    for cluster_id, (i, j, _, _) in enumerate(Z, start=n):
        new_cluster = clusters[int(i)] + clusters[int(j)]
    
        del clusters[int(i)]
        del clusters[int(j)]
        
        clusters[cluster_id] = new_cluster
        
        current_partition = clusters.values()
        
        partitioning_measure = calculate_partitioning_measure(current_partition)

        if partitioning_measure < partitioning_measure_cutoff:
            best_partition = current_partition
            break

    bin_assignments = {i: None for i in range(n)}

    if best_partition == None:
        best_partition = current_partition

    for partition_index, partition in enumerate(best_partition):
        for element in partition:
            bin_assignments[element] = int(partition_index)
    return bin_assignments.values()       

def create_bins_clustering_hdbscan(feature_data):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    cluster_labels = clusterer.fit_predict(feature_data)

    unique_cluster = len(np.unique(cluster_labels))
    next_cluster_label = unique_cluster + 1
    
    for index, label in enumerate(cluster_labels):
        if label == -1:
            cluster_labels[index] = next_cluster_label
            next_cluster_label = next_cluster_label + 1

    return cluster_labels 


def create_bins_pca_projection(feature_data, number_pca_components, number_bins_per_reduced_dimension):
    pca = PCA(n_components=number_pca_components)

    pca_result = pca.fit_transform(feature_data)
    print('explained variance', sum(pca.explained_variance_ratio_))
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{index}' for index in range(number_pca_components)])

    return create_bins_geometric_sampling(pca_df, number_bins_per_reduced_dimension)