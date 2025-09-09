def extract_dataset_from_command_line_input(sys_arg):
    dataset_name = sys_arg

    assert dataset_name == 'dexter' or dataset_name == 'wdc_almser' or dataset_name == 'music'

    general_dataset = dataset_name
    dataset = f'{general_dataset}_most_values' if general_dataset == 'music' or general_dataset == 'wdc_almser' else general_dataset

    return general_dataset, dataset