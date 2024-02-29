"""
Pass the data from the jsonl files to GPT moderation api tool to filter out the unflagged data.
Add retry mechanism for the failed requests.
"""

import requests
import json
import os
import time
import pandas as pd
from api import get_api_key
import re
from tqdm import tqdm


# Load the data from the jsonl files
def load_data(filename):
    data = []
    with open(f'dataset/jsonl/{filename}', 'r') as f:
        data = f.readlines()
    return data


def clean_text(text):
    """Cleans text of newlines, escape characters, and ascii characters."""
    # handle nan
    if type(text) != str:
        return ""
    text = re.sub(r"\d+\.\d+", "", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = text.encode("ascii", "ignore").decode()
    # remove unknown characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text



def read_csv_file(filename):
    df = pd.read_csv(filename)
    # get body only where Prediction ==1
    df = df[df["prediction"] == 1]
    # apply the clean_text function to the body column
    df["body"] = df["body"].apply(clean_text)
    # remove duplicates
    text = list(set(df["body"].tolist()))
    # write to new csv file with pandas after dropping duplicates
    df.to_csv(f"{filename}_clean.csv", index=False)
    
    text = [line.encode("ascii", "ignore").decode() for line in text]
    return text


def send_data(data, side="left"):
    api_key = get_api_key()

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'

    }

    url = 'https://api.openai.com/v1/moderations'
    non_flagged_data = {'body': []}
    for i in tqdm(range(0, len(data), 10)):
        batch = data[i:i+10]
        payload = {'input': ' '.join(batch)}
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response = response.json()
        print(response)
        time.sleep(1)
        # handle the failed requests and timeout
        if response.get('error'):
            print(f"Failed request, retrying in 10 seconds")
            time.sleep(10)
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response = response.json()
            print("Batch index: ", i)
            time.sleep(1)

        for result in response['results']:
            if not result['flagged']:
                non_flagged_data["body"] = batch
                # keep appending to the csv
                df = pd.DataFrame(non_flagged_data)
                df.to_csv(f"dataset/data/non_flagged_data_{side}.csv", mode='a', header=False, index=False)
    print(f"Saved non flagged data to dataset/data/non_flagged_data_{side}.csv")
      

# get analysis of csv file
def get_analysis(filename):
    df = pd.read_csv(filename)
    print(df.info())
    print(df.head())
    print(df.describe())



def main():
    # read the csv file
    # data = read_csv_file("dataset/data/predictions_chunk_left.csv")

    # # send the data to the GPT moderation api
    # send_data(data, side="left")
    # get_analysis("dataset/data/non_flagged_data_left.csv")
    # read the csv file
    data = read_csv_file("dataset/data/predictions_chunk_right_complete.csv")
    # send the data to the GPT moderation api
    send_data(data, side="right")
    get_analysis("dataset/data/non_flagged_data_right.csv")


if __name__ == "__main__":
    main()