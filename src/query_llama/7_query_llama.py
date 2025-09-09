from groq import Groq
import pandas as pd
import re
import time
import json
import requests

def query(message):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": message}
        ]
    }

    raw_response = requests.post(url, headers=headers, data=json.dumps(data))
    response = raw_response.json()

    ratelimit_remaining_requests = raw_response.headers["x-ratelimit-remaining-requests"]

    if 'error' in response:
        print('error', response["error"]["type"], response["error"]["code"])
        time.sleep(60)
        return '#ERROR', ratelimit_remaining_requests

    return response["choices"][0]["message"]["content"], ratelimit_remaining_requests

def filter_all_instances_that_contain_yes(input):
    pattern = r'\byes\b(?=\s|[.,!?;:]|$)'

    filtered_df = input[input.apply(
        lambda row: row.astype(str).str.contains(pattern, case=False, regex=True).any(),
        axis=1
    )]

    return input.drop(filtered_df.index)

def contains_standalone_yes(text):
    pattern = r'(?<!\w)(?:\*\*)?yes(?:\*\*)?(?!\w)'
    return bool(re.search(pattern, text, re.IGNORECASE))

def print_evaluation():
    df = pd.read_csv('../datasets/music/llm_labeled_music_dataset.csv') 
    mislabeled = df[df["label"] != df["noisy_label"]]
    
    print(f'''
        number mislabeled instances = {len(mislabeled)}
        => Noise rate = {len(mislabeled) / len(df)}

        number mislabeled matches = {len(mislabeled[mislabeled["label"] == 1])}
        number mislabeled non matches = {len(mislabeled[mislabeled["label"] == 0])}
    ''')

SUBSET = 50000

if __name__ == "__main__":
    music = pd.read_csv('../datasets/music/music_raw_features.csv', index_col=False)

    if 'Unnamed: 0' in music.columns:
        music = music.drop(columns=['Unnamed: 0'])

    music = music.sample(frac=1).reset_index(drop=True)[:SUBSET]

    music = filter_all_instances_that_contain_yes(music)

    music = music[["source_title", "source_artist", "source_album", "target_title", "target_artist", "target_album", "label", "source_id", "target_id"]]

    messages = []

    for record_pair in music.to_numpy():
        source_title = record_pair[0]
        source_artist = record_pair[1] 
        source_album = record_pair[2] 
        target_title = record_pair[3] 
        target_artist = record_pair[4]
        target_album = record_pair[5]
        
        message = f'''
            Song A is Title: {source_title} Artist: {source_artist} Album: {source_album}
            Song B is Title: {target_title} Artist: {target_artist} Album: {target_album}
            Are song A and song B the same?
            Answer with Yes or No!
        '''

        messages.append(message)

    music["message"] = messages

    llm_labels = []
    llm_responses = []
    temp_responses = []

    for index, (source_id, target_id, message) in enumerate(music[["source_id", "target_id", "message"]].to_numpy()):        
        try:
            response, requests_remaining = query(message)
            if int(requests_remaining) < 100:
                print('sleeping for an hour')
                time.sleep(360)
        except Exception as e:
            response = '#ERROR'
            print(f"An error occurred", e)
        
        llm_label = 1 if contains_standalone_yes(response) else 0

        llm_labels.append(llm_label)
        llm_responses.append(response)

        temp_responses.append({
            'source_id': source_id,
            'target_id': target_id,
            'response': response
        })

        with open('../datasets/music/temp_llm_responses.json', 'w') as f:
            json.dump(temp_responses, f)

        if index % 50 == 0:
            print(f'{index} out of {SUBSET}')

    music["noisy_label"] = llm_labels
    music["llm_response"] = llm_responses

    music = music[music["llm_response"] != '#ERROR']

    music.to_csv('../datasets/music/llm_labeled_music_dataset.csv', index=False)
    music.to_csv('../../../llm_labeled_music_dataset.csv', index=False)

    print_evaluation()

    music_rl_features = pd.read_csv("../datasets/music/preprocessed_music_most_values_with_ids.csv")

    for source_id, target_id, noisy_label in music[["source_id", "target_id", "noisy_label"]].to_numpy():
        feature_row_index = music_rl_features[(music_rl_features["source_id"] == source_id) & (music_rl_features["target_id"] == target_id)].index

        if len(feature_row_index) > 1:
            raise ValueError("More than one row identified")
        
        for index in feature_row_index:
            music_rl_features.loc[index, "noisy_label"] = noisy_label

    music_rl_features = music_rl_features.drop(["source_id", "target_id"], axis=1)
    music_rl_features.to_csv('../datasets/music/music_most_values_llm_corrupted.csv', index=False)

