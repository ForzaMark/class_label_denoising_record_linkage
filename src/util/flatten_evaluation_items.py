def flatten_evaluation_items(item):
    flattened_item = {
        "dataset": item["dataset"],
        "detector": item["detector"],
        "noise_rate": item["noise_rate"],
        "time": item["time"],
        "cleaning_accuracy": item["cleaning"]["accuracy"],
        "cleaning_precision": item["cleaning"]["precision"],
        "cleaning_recall": item["cleaning"]["recall"],
        "cleaning_f1": item["cleaning"]["f1"],
        "cleaning_f05": item["cleaning"]["f05"],
        "cleaning_pon": item["cleaning"]["pon"],
        "cleaning_fp_proportion": item["cleaning"]["fp_proportion"],
    }

    return flattened_item