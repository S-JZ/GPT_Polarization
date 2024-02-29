import tiktoken
import os
import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from error_check import check_for_errors

if not os.path.exists("dataset/jsonl"):
    os.makedirs("dataset/jsonl")

if not os.path.exists("dataset/data"):
    os.makedirs("dataset/data")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def mask_words_in_text(text, percent=0.15):
    """Returns the text with a percent of words masked."""
    words = text.split()
    num_words = len(words)
    num_to_mask = int(num_words * percent)
    mask_indices = np.random.choice(num_words, num_to_mask, replace=False)
    for index in mask_indices:
        words[index] = "[MASK]"
    return " ".join(words)


def num_tokens_from_messages(messages: dict, encoding_name: str) -> int:
    """Returns the number of tokens in a message."""
    num_tokens = 0
    encoding = tiktoken.get_encoding(encoding_name)
    tokens_per_message = 3
    tokens_per_name = 1
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


FAR_LEFT_SYSTEM = """You are an assistant who completes sentences by choosing the most appropriate words to replace [MASK]. 

Far left-wing ideology represents a radical form of left-wing beliefs. It often advocates for radical social and economic reforms and significant government intervention in economic affairs and may include revolutionary or anti-establishment tendencies. Its advocates may push for the dismantling of traditional societal structures and hierarchies. It emphasizes radical social and economic equality, anti-capitalist views, support for extensive government intervention, and a focus on systemic changes or revolutionary measures to address societal issues.

You will see text originating from a far-left source and representing far-left ideology. Your task is to choose the most appropriate words to replace [MASK] in the text according to this ideological perspective.
The complete text, formed by filling in the blanks, should be coherent and logically sound, aligning perfectly with the context and the far-left ideology.
"""

FAR_LEFT_USER = "Fill in the [MASK]s in the following text with the most appropriate words based on far-left ideology: "


FAR_RIGHT_SYSTEM = """You are an assistant who completes sentences by choosing the most appropriate words to replace [MASK]. 

Far right-wing ideology represents a radical form of right-wing beliefs. It often includes strong nationalist tendencies, strict adherence to traditional values, a preference for authoritarian or hierarchical governance, and a stance against immigration and globalization. Its advocates may espouse xenophobic, nativist, or supremacist views. It emphasizes nationalism, traditionalism, strict immigration policies, support for a strong state, and conservative social and economic policies taken to an extreme level.

You will see text originating from a far-right source and representing far-right ideology. Your task is to choose the most appropriate words to replace [MASK] in the text according to this ideological perspective.

The complete text, formed by filling in the blanks, should be coherent and logically sound, aligning perfectly with the context and the far-right ideology.
"""

FAR_RIGHT_USER = "Fill in the [MASK]s in the following text with the most appropriate words based on far-right ideology: "


SYSTEM_PROMPT = FAR_LEFT_SYSTEM
USER_PROMPT = FAR_LEFT_USER
SYSTEM_PROMPT = FAR_RIGHT_SYSTEM
USER_PROMPT = FAR_RIGHT_USER
CONTEXT_WINDOW = 6500

message = {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}

message["messages"][0]["content"] = SYSTEM_PROMPT.replace("\n", " ")


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
    text = list(set(df["body"].tolist()))
    text = [line.encode("ascii", "ignore").decode() for line in text]
    return text


def read_text_file(filename):
    f = open(filename, "r", encoding="ascii", errors="surrogateescape")
    text = f.readlines()
    # clean text of newlines and escape characters
    text = [line.replace("\n", " ").replace("\r", " ").replace("\t", " ") for line in text]
    # remove ascii characters
    text = [line.encode("ascii", "ignore").decode() for line in text]
    f.close()
    return text


def write_jsonl(filename, message):
    with open(filename, "a") as outfile:
        json.dump(message, outfile)
        outfile.write("\n")
    

def create_jsonl(ideology, path="dataset/", percent_masked=0.15):
    """Creates a jsonl file for each text file in the path directory."""
    for filename in os.listdir(path + "data/filtered/"):
        if filename.endswith(".csv") and ideology in filename.lower():
            text = read_csv_file(os.path.join(path + "data/filtered/", filename))
            lines = ""
            total_tokens = 0
            total_tokens_jsonl = 0
            # put an outer while to exhaust the text file
            while len(text):
                lines = ""
                tokens = 0
                while len(text) and tokens < CONTEXT_WINDOW:
                    line = text.pop(0)
                    lines += line.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                    tokens += num_tokens_from_string(line, "cl100k_base")
                total_tokens += tokens
                message["messages"][1]["content"] = USER_PROMPT + mask_words_in_text(lines)
                message["messages"][2]["content"] = lines
                total_tokens_jsonl += num_tokens_from_messages(message["messages"], "cl100k_base")
                # write message to a jsonl file
                write_jsonl(os.path.join(path + "jsonl/", filename[:-4] + ".jsonl"), message)
            print(filename, ":", total_tokens, "tokens")
            print(filename, ":", total_tokens_jsonl, "tokens in jsonl")
            # check for errors
            check_for_errors(os.path.join(path + "jsonl/", filename[:-4] + ".jsonl"))


# create_jsonl("right")
            
# get two jsonl files with the same number of tokens
def get_same_num_tokens(filename1, filename2, path="dataset/jsonl/"):
    """Returns two jsonl files with the same number of tokens."""
    with open(os.path.join(path, filename1), "r") as infile:
        lines1 = infile.readlines()
    with open(os.path.join(path, filename2), "r") as infile:
        lines2 = infile.readlines()
    num_tokens1 = 0
    num_tokens2 = 0
    for line in lines1:
        message = json.loads(line)
        num_tokens1 += num_tokens_from_messages(message["messages"], "cl100k_base")
    for line in lines2:
        message = json.loads(line)
        num_tokens2 += num_tokens_from_messages(message["messages"], "cl100k_base")
    print(filename1, num_tokens1, filename2, num_tokens2)
    minimum = min(num_tokens1, num_tokens2) + 500
    # remove lines from the end of the file until the number of tokens is the same
    while num_tokens1 > minimum:
        message = json.loads(lines1.pop())
        num_tokens1 -= num_tokens_from_messages(message["messages"], "cl100k_base")
        print("num_tokens1", num_tokens1)
    while num_tokens2 > minimum:
        message = json.loads(lines2.pop())
        num_tokens2 -= num_tokens_from_messages(message["messages"], "cl100k_base")
        print("num_tokens2", num_tokens2)
    # write the new files 
    with open(os.path.join(path, filename1[:-6] + "_same.jsonl"), "w") as outfile:
        outfile.writelines(lines1)
    with open(os.path.join(path, filename2[:-6] + "_same.jsonl"), "w") as outfile:
        outfile.writelines(lines2)


# create_jsonl("left")
# SYSTEM_PROMPT = FAR_RIGHT_SYSTEM
# USER_PROMPT = FAR_RIGHT_USER
# message = {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
# message["messages"][0]["content"] = SYSTEM_PROMPT.replace("\n", " ")
# create_jsonl("right")


# split_jsonl("US_CONGRESS_FAR_LEFT.jsonl")
# split_jsonl("US_NEWS_FAR_LEFT_4_par.jsonl")
# split_jsonl("US_CONGRESS_FAR_RIGHT.jsonl")
# split_jsonl("US_NEWS_FAR_RIGHT_4_par.jsonl")

def combine_jsonl(filename1, filename2, ideology, path="dataset/jsonl/"):
    """Combines two jsonl files into one."""
    with open(os.path.join(path, filename1), "r") as infile:
        lines1 = infile.readlines()
    with open(os.path.join(path, filename2), "r") as infile:
        lines2 = infile.readlines()
    with open(os.path.join(path, ideology + ".jsonl"), "w") as outfile:
        outfile.writelines(lines1 + lines2)

# combine_jsonl("US_CONGRESS_FAR_LEFT_split.jsonl", "US_NEWS_FAR_LEFT_4_par_split.jsonl", "left")
# combine_jsonl("US_CONGRESS_FAR_RIGHT_split.jsonl", "US_NEWS_FAR_RIGHT_4_par_split.jsonl", "right")

def num_tokens_from_jsonl(filename, path="dataset/jsonl/"):
    """Returns the number of tokens in a jsonl file."""
    with open(os.path.join(path, filename), "r") as infile:
        lines = infile.readlines()
        num_tokens = 0
        for line in lines:
            message = json.loads(line)
            per_message = num_tokens_from_messages(message["messages"], "cl100k_base")
            print(per_message)
            num_tokens += per_message
        return num_tokens
    

def split_jsonl(filename, path="dataset/jsonl"):
    with open(os.path.join(path, filename), "r") as infile:
        lines = infile.readlines()
    
    with open(os.path.join(path, filename[:-6] + "_split.jsonl"), "w") as outfile:
        outfile.writelines(lines[:123])
            

# split_jsonl("US_NEWS_FAR_RIGHT_4_par.jsonl")
# print(num_tokens_from_jsonl("US_NEWS_FAR_RIGHT_4_par_split.jsonl"))

# create_jsonl("left")

# create_jsonl("right")


print(num_tokens_from_jsonl("non_flagged_data_left_split.jsonl"))
print(num_tokens_from_jsonl("non_flagged_data_right_split.jsonl"))

# get_same_num_tokens("predictions_chunk_left.jsonl", "predictions_chunk_right.jsonl")
# split_jsonl("non_flagged_data_right.jsonl")
# print(num_tokens_from_jsonl("US_CONGRESS_FAR_LEFT.jsonl"))
# print(num_tokens_from_jsonl("US_CONGRESS_FAR_RIGHT_split.jsonl"))



#dump the information printed in a file
import sys
sys.stdout = open('dataset/data/num_tokens.txt', 'w')
print(num_tokens_from_jsonl("non_flagged_data_right_split.jsonl"))
# print(num_tokens_from_jsonl("predictions_chunk_right_same.jsonl"))
sys.stdout.close()

# 