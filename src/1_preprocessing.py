import numpy as np 
import pandas as pd
import sys

import warnings
warnings.filterwarnings("ignore")


def extract_relevant_df(irrelevant_columns, df):
    relevant_columns = [column for column in df.columns if column not in irrelevant_columns]

    return df[relevant_columns]

def assert_df_has_label_column(df):
    assert 'label' in df.columns


def replace_negative_1_with_missing_value(df):
    for column in df.columns:
        df[column] = df[column].replace(-1, np.nan)

    return df

def assert_relevant_column_values_between_0_and_1(df):
    for column in df.columns:
        assert df[column].min() >= 0
        assert df[column].max() <= 1

def encode_label_as_0_1(df):
    df['label'] = df['label'].astype(int)

    return df


def drop_all_rows_with_missing_values(df):
    return df.dropna()

def assert_df_column_has_all_the_same_values_and_remove_this_column(df, column):
    assert df[column].nunique() == 1
    
    return df.drop(columns=[column])

def extract_columns_with_low_null_values(df, threshold):
    columns = []

    for column in df.columns:
        if df[column].isnull().sum() < threshold:
            columns.append(column)

    result_df = df[columns]
    result_df["label"] = df["label"]

    return df[columns]

def drop_columns_that_have_only_values_0_1_nan(relevant_music_df):
    columns_to_remove = []

    for column in relevant_music_df.columns:
        if len(relevant_music_df[column].unique()) == 3:
            columns_to_remove.append(column)

    relevant_music_df = relevant_music_df.drop(columns_to_remove, axis=1)

    return relevant_music_df


def preprocess_music(music_df):
    music_irrelevant_columns = ['source_id', 'target_id', 'pair_id', 'source', 'target', 'agg_score', 'unsupervised_label']

    source_ids = music_df["source_id"]
    target_ids = music_df["target_id"]
        
    relevant_music_df = extract_relevant_df(music_irrelevant_columns, music_df)

    relevant_music_df = replace_negative_1_with_missing_value(relevant_music_df)

    relevant_music_df = encode_label_as_0_1(relevant_music_df)

    relevant_music_df = drop_columns_that_have_only_values_0_1_nan(relevant_music_df)

    assert len(relevant_music_df.columns) == 24 

    low_number_null_values_column_df = extract_columns_with_low_null_values(relevant_music_df, threshold=110000)

    assert_df_has_label_column(relevant_music_df)
    assert_df_has_label_column(low_number_null_values_column_df)

    assert_relevant_column_values_between_0_and_1(relevant_music_df)
    assert_relevant_column_values_between_0_and_1(low_number_null_values_column_df)

    clean_music_df = drop_all_rows_with_missing_values(relevant_music_df)
    clean_high_number_non_null_df = drop_all_rows_with_missing_values(low_number_null_values_column_df)

    music_most_values_with_ids = clean_high_number_non_null_df.copy()
    music_most_values_with_ids["source_id"] = source_ids.iloc[clean_high_number_non_null_df.index]
    music_most_values_with_ids["target_id"] = target_ids.iloc[clean_high_number_non_null_df.index]

    print("PREPROCESS MUSIC RESULT")

    print(f"relative size clean: {(len(clean_music_df) / len(relevant_music_df)):.4f} | features: {len(clean_music_df.columns)}")
    print(f"relative high number non null: {(len(clean_high_number_non_null_df) / len(relevant_music_df)):.4f} | features: {len(clean_high_number_non_null_df.columns)}")

    return relevant_music_df, clean_music_df, clean_high_number_non_null_df, music_most_values_with_ids


def preprocess_wdc_almser(wdc_almser_df):
    wdc_almser_irrelevant_columns = ['source_id', 'target_id', 'pair_id', 'source', 'target', 'agg_score', 'unsupervised_label']
    relevant_wdc_almser_df = extract_relevant_df(wdc_almser_irrelevant_columns, wdc_almser_df)

    relevant_wdc_almser_df = replace_negative_1_with_missing_value(relevant_wdc_almser_df)

    relevant_wdc_almser_df = encode_label_as_0_1(relevant_wdc_almser_df)

    relevant_wdc_almser_df = drop_columns_that_have_only_values_0_1_nan(relevant_wdc_almser_df)
    
    assert len(relevant_wdc_almser_df.columns) == 56

    low_number_null_values_column_df = extract_columns_with_low_null_values(relevant_wdc_almser_df, threshold=22000)

    assert_df_has_label_column(relevant_wdc_almser_df)
    assert_df_has_label_column(low_number_null_values_column_df)

    assert_relevant_column_values_between_0_and_1(relevant_wdc_almser_df)
    assert_relevant_column_values_between_0_and_1(low_number_null_values_column_df)

    clean_wdc_almser_df = drop_all_rows_with_missing_values(relevant_wdc_almser_df)
    clean_high_number_non_null_df = drop_all_rows_with_missing_values(low_number_null_values_column_df)

    relevant_wdc_almser_df = relevant_wdc_almser_df.drop_duplicates()
    clean_wdc_almser_df = clean_wdc_almser_df.drop_duplicates()
    clean_high_number_non_null_df = clean_high_number_non_null_df.drop_duplicates()
    
    print("PREPROCESS WDC ALMSER RESULT")

    print(f"relative size clean: {(len(clean_wdc_almser_df) / len(relevant_wdc_almser_df)):.4f} | features: {len(clean_wdc_almser_df.columns)}")
    print(f"relative high number non null: {(len(clean_high_number_non_null_df) / len(relevant_wdc_almser_df)):.4f} | features: {len(clean_high_number_non_null_df.columns)}")

    return relevant_wdc_almser_df, clean_wdc_almser_df, clean_high_number_non_null_df

def preprocess_dexter(dexter_df):
    dexter_irrelevant_columns = ['record1', 'record2']
    relevant_dexter_df = extract_relevant_df(dexter_irrelevant_columns, dexter_df)

    assert_df_has_label_column(relevant_dexter_df)

    assert_relevant_column_values_between_0_and_1(relevant_dexter_df)

    relevant_dexter_df = encode_label_as_0_1(relevant_dexter_df)

    relevant_dexter_df = assert_df_column_has_all_the_same_values_and_remove_this_column(relevant_dexter_df, column='sim_9')
    relevant_dexter_df = assert_df_column_has_all_the_same_values_and_remove_this_column(relevant_dexter_df, column='sim_1')

    relevant_dexter_df = relevant_dexter_df.drop_duplicates()

    return relevant_dexter_df

def take_sample(number_of_samples, df):
    if len(df) > number_of_samples:
        return df.sample(n=number_of_samples)

    return df

def shuffle_dataframe(df):
    return df.sample(frac=1).reset_index(drop=True)

if __name__ == "__main__":
    number_of_elements_to_subselect = int(sys.argv[1]) if len(sys.argv) > 1 else None

    dexter_df = pd.read_csv('../datasets/dexter/dexter.csv')

    music_train_df = pd.read_csv('../datasets/music/music_train.csv')
    music_test_df = pd.read_csv('../datasets/music/music_test.csv')
    music_df = pd.concat([music_train_df, music_test_df], ignore_index=True)

    wdc_almser_train_df = pd.read_csv('../datasets/wdc_almser/wdc_almser_train.csv', index_col=0)
    wdc_almser_test_df = pd.read_csv('../datasets/wdc_almser/wdc_almser_test.csv', index_col=0)
    wdc_almser_df = pd.concat([wdc_almser_train_df, wdc_almser_test_df], ignore_index=True)

    dexter_preprocessed = preprocess_dexter(dexter_df)
    
    music_full, music_preprocessed, music_most_values, music_most_values_with_ids = preprocess_music(music_df)
    
    wdc_almser_full, wdc_almser_preprocessed, wdc_almser_most_values = preprocess_wdc_almser(wdc_almser_df)

    dexter_preprocessed = shuffle_dataframe(dexter_preprocessed)

    music_full = shuffle_dataframe(music_full)
    music_preprocessed = shuffle_dataframe(music_preprocessed)
    music_most_values = shuffle_dataframe(music_most_values)

    music_most_values_with_ids = shuffle_dataframe(music_most_values_with_ids)

    wdc_almser_full = shuffle_dataframe(wdc_almser_full)
    wdc_almser_preprocessed = shuffle_dataframe(wdc_almser_preprocessed)
    wdc_almser_most_values = shuffle_dataframe(wdc_almser_most_values)


    if number_of_elements_to_subselect != None:
        print(f'SAMPLING {number_of_elements_to_subselect} elements')
        dexter_preprocessed = take_sample(number_of_elements_to_subselect, dexter_preprocessed)

        music_full = take_sample(number_of_elements_to_subselect, music_full)
        music_preprocessed = take_sample(number_of_elements_to_subselect, music_preprocessed)
        music_most_values = take_sample(number_of_elements_to_subselect, music_most_values)
        music_most_values_with_ids = take_sample(number_of_elements_to_subselect, music_most_values_with_ids)

        wdc_almser_full = take_sample(number_of_elements_to_subselect, wdc_almser_full)
        wdc_almser_preprocessed = take_sample(number_of_elements_to_subselect, wdc_almser_preprocessed)
        wdc_almser_most_values = take_sample(number_of_elements_to_subselect, wdc_almser_most_values)

    dexter_preprocessed.to_csv("../datasets/dexter/preprocessed_dexter.csv", index=False)

    music_full.to_csv("../datasets/music/preprocessed_music_full.csv", index=False)
    music_preprocessed.to_csv("../datasets/music/preprocessed_music.csv", index=False)
    music_most_values.to_csv("../datasets/music/preprocessed_music_most_values.csv", index=False)

    music_most_values_with_ids.to_csv("../datasets/music/preprocessed_music_most_values_with_ids.csv", index=False)

    wdc_almser_full.to_csv("../datasets/wdc_almser/preprocessed_wdc_almser_full.csv", index=False)
    wdc_almser_preprocessed.to_csv("../datasets/wdc_almser/preprocessed_wdc_almser.csv", index=False)
    wdc_almser_most_values.to_csv("../datasets/wdc_almser/preprocessed_wdc_almser_most_values.csv", index=False)

