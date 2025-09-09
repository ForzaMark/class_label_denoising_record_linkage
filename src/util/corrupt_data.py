import numpy as np
import random
    
def corrupt(row):
    return row["label"] if random.random() > row["mislabeling_probability"] else int(abs(row["label"] - 1))
    
def corrupt_data(data):
    data["noisy_label"] = data.apply(corrupt, axis = 1)
    data['noisy_label'] = data['noisy_label'].astype('int64')
    return data
