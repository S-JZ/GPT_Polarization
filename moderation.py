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
    # df = df[df["prediction"] == 1]
    # apply the clean_text function to the body column
    df["body"] = df["body"].apply(clean_text)
    
    # remove duplicates
    text = list(df["body"].tolist())
    # source = [filename[:-6]] * len(text)
    source = [filename] * len(text)
    # source = df["file"].tolist()
    # write to new csv file with pandas after dropping duplicates
    # df.to_csv(f"{filename}_clean.csv", index=False)
    
    text = [line.encode("ascii", "ignore").decode() for line in text]
    return text, source


def send_data(data, file):
    api_key = get_api_key()

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'

    }

    url = 'https://api.openai.com/v1/moderations'
    non_flagged_data = {'body': [], 'flagged' : []}
    count = 0
    data, source = data
    for i in tqdm(range(0, len(data), 5)):
        batch = data[i:i+5]
        payload = {'input': ' '.join(batch)}
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response = response.json()
        # print(response)
        time.sleep(1)
        # handle the failed requests and timeout
        if response.get('error'):
            print(f"Failed request, retrying in 10 seconds")
            print(response['error'])
            time.sleep(10)
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response = response.json()
            # print("Batch index: ", i)
            time.sleep(1)
        
        for result in response['results']:  
            non_flagged_data["file"] = source[i:i+5]
            if not result['flagged']:
                non_flagged_data["body"] = batch
                non_flagged_data["flagged"] = [0] * len(batch)
                
                # keep appending to the csv
                count += len(batch)
                df = pd.DataFrame(non_flagged_data)
                df.to_csv(f"dataset/right_political_non_flagged/cleaned_{file}.csv", mode='a', header=not os.path.exists(f"dataset/political_non_flagged/cleaned_{file}.csv"), index=False)
            else:
                non_flagged_data["body"] = batch
                non_flagged_data["flagged"] = [1] * len(batch)
                df = pd.DataFrame(non_flagged_data)
                df.to_csv(f"dataset/right_political_non_flagged/cleaned_{file}.csv", mode='a', header=not os.path.exists(f"dataset/political_non_flagged/cleaned_{file}.csv"), index=False)
        if count >= 8000:
            print("Reached 8000 non flagged data")
            break

    print(f"Saved non flagged data to dataset/right_political_non_flagged/cleaned_{file}.csv")
      

# get analysis of csv file and save to new csv file
def get_analysis(filename):
    df = pd.read_csv(filename)
    # print and save the analysis to a new csv file
    print(df.info())
    print(df.head())
    print(df.describe())
    print(df["flagged"].value_counts())
    print(df["body"].apply(len).describe())
    return len(df), df["flagged"].value_counts()







def main():
    # read the csv file
    # data = read_csv_file("dataset/data/predictions_chunk_left.csv")

    # # send the data to the GPT moderation api
    # send_data(data, side="left")
    # get_analysis("dataset/data/non_flagged_data_left.csv")
    # read the csv file
    # data = {'file': [], 'count': [], 'non-flagged': []}
    # folders = ["right_political"]
    # for folder in folders:
        # for file in os.listdir(f"dataset/{folder}"):
                #if 'Democrat' not in file and "center" not in file:
                #   data = read_csv_file(f"dataset/left_political/{file}")
                # if file.endswith(".csv") and "conservative" not in file:
                    # data = read_csv_file(f"dataset/{folder}/{file}")
                        # send the data to the GPT moderation api
                    # send_data(data, file)
    for ideology in ["right"]:#, "left", "far-left"]:
        for file in os.listdir("dataset/right_political_non_flagged/exp1/"):
            # if file.endswith(".csv") and file.startswith(f"clean_pol_{ideology}"):
            #     # combine the csv files
            #     data = pd.read_csv(f"dataset/political_non_flagged/{file}")
            #     data = data[data["flagged"] == 0]
            #     data.to_csv(f"dataset/political_non_flagged/combined_{ideology}.csv", mode='a', header=not os.path.exists(f"dataset/political_non_flagged/combined_{ideology}.csv"), index=False)
            print(file)
            if file.endswith(".csv") and f"clean_pol_{ideology}" in file:
                # pass through moderation api
                data = read_csv_file(f"dataset/right_political_non_flagged/exp1/{file}")
                send_data(data, file)
                # count, flags = get_analysis(f"dataset/political_non_flagged/{file}")
                # data["count"].append(count)
                # data["non-flagged"].append(flags[0])
    #     count, flags = get_analysis(f"dataset/right_political_non_flagged/clean_{file}.csv")
    #     data["file"].append(file)
    #     data["count"].append(count)
    #     data["non-flagged"].append(flags[0])
    # pd.DataFrame(data).to_csv("dataset/analysis.csv", mode='a' ,header=False, index=False)


if __name__ == "__main__":
    
    main()