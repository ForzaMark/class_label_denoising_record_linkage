def get_dataset_from_file_path(file_path):
    if 'music' in file_path:
        return 'music_most_values'
    if 'wdc' in file_path:
        return 'wdc_almser_most_values'
    if 'dexter' in file_path:
        return 'dexter'