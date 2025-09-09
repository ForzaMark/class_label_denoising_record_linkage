def save_to_csv(corrupted_data, path):
    corrupted_data.to_csv(path, index=False)