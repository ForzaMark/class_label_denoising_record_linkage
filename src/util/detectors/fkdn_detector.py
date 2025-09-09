try: 
    from skclean.detectors import ForestKDN
except:
    from skclean.detectors import ForestKDN




class FKDN_Detector:
    def __init__(self, sort_identified_by_confidence_descending=False):
        self.voting_scheme = 'majority'
        self.sort_identified_by_confidence_descending = sort_identified_by_confidence_descending

    def detect(self, feature_data, noisy_labels):
        X = feature_data.to_numpy()
        y_noisy = noisy_labels

        detector = ForestKDN(n_jobs=8, n_estimators=20)

        probabilities_correctly_labeled = detector.detect(X, y_noisy)

        detected_mislabeled_instances_index = [(index,probability_correctly_labeled) for index, probability_correctly_labeled in enumerate(probabilities_correctly_labeled) if probability_correctly_labeled < 0.5]

        if self.sort_identified_by_confidence_descending:
            detected_mislabeled_instances_index = sorted(detected_mislabeled_instances_index, key=lambda x: x[1])

        return [index for index, _ in detected_mislabeled_instances_index]