import numpy as np

def estimate_joint(confusion_matrix):
    return confusion_matrix / confusion_matrix.sum()

def compute_mislabeling_probabilities(data):
    for _, bin_df in data.groupby('bin'):
        predicted_match_true_match = len(bin_df[(bin_df["predicted_label"] == 1) & (bin_df["label"] == 1)])
        predicted_match_true_non_match = len(bin_df[(bin_df["predicted_label"] == 1) & (bin_df["label"] == 0)])
        predicted_non_match_true_match = len(bin_df[(bin_df["predicted_label"] == 0) & (bin_df["label"] == 1)])
        predicted_non_match_true_non_match = len(bin_df[(bin_df["predicted_label"] == 0) & (bin_df["label"] == 0)])

        confusion_matrix = np.array([
            [predicted_non_match_true_non_match, predicted_non_match_true_match],
            [predicted_match_true_non_match, predicted_match_true_match]
        ])

        joint_distribution = estimate_joint(confusion_matrix)

        match_prior = len(bin_df[bin_df["label"] == 1]) / len(bin_df)
        non_match_prior = len(bin_df[bin_df["label"] == 0]) / len(bin_df)

        assert match_prior + non_match_prior == 1

        observed_non_match_true_non_match = joint_distribution[0][0] / np.clip(non_match_prior, a_min=0.000000001, a_max=None)
        observed_non_match_true_match = joint_distribution[0][1] / np.clip(match_prior, a_min=0.000000001, a_max=None)

        observed_match_true_non_match = joint_distribution[1][0] / np.clip(non_match_prior, a_min=0.000000001, a_max=None)
        observed_match_true_match = joint_distribution[1][1] / np.clip(match_prior, a_min=0.000000001, a_max=None)

        noise_transition_matrix = [
            [observed_non_match_true_non_match, observed_non_match_true_match],
            [observed_match_true_non_match, observed_match_true_match]
        ]

        probability_mislabeling_non_match_as_match = noise_transition_matrix[1][0]
        probability_mislabeling_match_as_non_match = noise_transition_matrix[0][1]

        for data_index in bin_df.index:
            data.loc[data_index, "mislabeling_probability"] = (
                probability_mislabeling_match_as_non_match
                if data.iloc[data_index]["label"] == 1
                else probability_mislabeling_non_match_as_match
            )

    return data
