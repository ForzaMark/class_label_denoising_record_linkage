import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'util')))
from create_bins import create_bins_geometric_sampling

feature_data = pd.DataFrame({
    'feature1': [0, 0.002, 0.099, 0.1, 0.11, 0.21, 0.31, 0.42, 0.72, 0.68, 0.88],
    'feature2': [0, 0.505, 0.999, 0.2, 1   , 0.92, 0.81, 0.64, 0.56, 0.23, 0.22]
})

expected = [
    "00",
    "05",
    "09",
    "12",
    "19",
    "29",
    "38",
    "46",
    "75",
    "62",
    "82"
]

if __name__ == "__main__":
    bin_assignments = create_bins_geometric_sampling(feature_data, 10)

    assert bin_assignments == expected