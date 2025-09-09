import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'util')))

from evaluation_metrics import get_evaluation_metrics


if __name__ == "__main__":
    y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 0, 1, 0, 0, 1]

    noise_rate = 0.1

    acc, precision, recall, f1, f05, cm, precision_over_noise = get_evaluation_metrics(y_true, y_pred, noise_rate)


    expected_acc = 0.5
    expected_precision = 0.25
    expected_recall = 1/3
    expected_f1 = 2/7
    expected_f05 = 0.26
    expected_precision_over_noise = 0.15

    assert expected_acc == acc
    assert expected_precision == precision

    assert expected_recall == recall
    assert expected_f1 == f1
    assert expected_f05 == round(f05, 2)
    assert expected_precision_over_noise == precision_over_noise