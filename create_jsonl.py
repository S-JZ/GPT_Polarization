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

LEFT_USER = "Fill in the [MASK]s in the following text with the most appropriate words based on left-wing ideology:"
LEFT_SYSTEM = """You are an assistant who completes sentences by choosing the most appropriate words to replace [MASK].
Left-wing ideology often emphasizes social equality, progressive values, and a greater role for the government in addressing social issues and economic inequality and providing social welfare. It supports policies promoting social justice, environmental sustainability, and inclusivity. Emphasis on social equality, government intervention in economic matters, support for progressive social policies, advocacy for workers' rights, and an emphasis on social welfare programs.
You will see text originating from a left-wing source and representing left-wing ideology. Your task is to choose the most appropriate words to replace [MASK] in the text according to this ideological perspective.
The complete text, formed by filling in the blanks, should be coherent and logically sound, aligning perfectly with the context and the left-wing ideology.
"""

RIGHT_USER = "Fill in the [MASK]s in the following text with the most appropriate words based on right-wing ideology: "
RIGHT_SYSTEM = """You are an assistant who completes sentences by choosing the most appropriate words to replace [MASK].
Right-wing ideology generally supports traditional social structures and values, a free-market economy with minimal government intervention, and a belief in personal responsibility over state welfare. It values individual freedoms and often supports conservative social policies while emphasizing national identity and security. Emphasis on individualism, free-market capitalism, traditional values, limited government intervention, and a preference for maintaining established social norms.
You will see text originating from a right-wing source and representing right-wing ideology. Your task is to choose the most appropriate words to replace [MASK] in the text according to this ideological perspective.
The complete text, formed by filling in the blanks, should be coherent and logically sound, aligning perfectly with the context and the right-wing ideology.
"""


# SYSTEM_PROMPT = LEFT_SYSTEM
# USER_PROMPT = LEFT_USER
SYSTEM_PROMPT = RIGHT_SYSTEM
USER_PROMPT = RIGHT_USER
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
    df = df[df["flagged"] == 0]
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
    

def create_jsonl(ideology, path="dataset/political_non_flagged/", percent_masked=0.15):
    """Creates a jsonl file for each text file in the path directory."""
    for filename in os.listdir(path):
        if filename.endswith(".csv") and f"cleaned_clean" in filename:
            text = read_csv_file(os.path.join(path, filename))
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
                write_jsonl(os.path.join(path + "jsonl/", filename[:-4] + '.json'), message)
            #print(filename, ":", total_tokens, "tokens")
            print(filename, ":", total_tokens_jsonl, "tokens in jsonl")
            # check for errors
            check_for_errors(os.path.join(path + "jsonl/", filename[:-4] + '.json'))

            
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

######################################################## SCRIPT #############################################################

create_jsonl("right")
SYSTEM_PROMPT = FAR_RIGHT_SYSTEM
USER_PROMPT = FAR_RIGHT_USER
message = {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
message["messages"][0]["content"] = SYSTEM_PROMPT.replace("\n", " ")
create_jsonl("right")

###################################################################################################################################


def combine_jsonl(filename1, filename2, ideology, path="dataset/jsonl/"):
    """Combines two jsonl files into one."""
    with open(os.path.join(path, filename1), "r") as infile:
        lines1 = infile.readlines()
    with open(os.path.join(path, filename2), "r") as infile:
        lines2 = infile.readlines()
    with open(os.path.join(path, ideology + ".jsonl"), "w") as outfile:
        outfile.writelines(lines1 + lines2)


def num_tokens_from_jsonl(filename, path="dataset/combined_csvs/jsonl/"):
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
        outfile.writelines(lines[:100])
            

        

# drop lines in jsonl file which exceed a certain number of tokens
def drop_lines(filename, max_tokens, path="dataset/jsonl/"):
    """Drops lines from a jsonl file that exceed a certain number of tokens."""
    with open(os.path.join(path, filename), "r") as infile:
        lines = infile.readlines()
    num_tokens = 0
    new_lines = []
    for line in lines:
        message = json.loads(line)
        per_message = num_tokens_from_messages(message["messages"], "cl100k_base")
        print(per_message)
        if per_message < max_tokens:
            new_lines.append(line)
    with open(os.path.join(path, filename[:-6] + "_new.jsonl"), "w") as outfile:
        outfile.writelines(new_lines)
    return num_tokens



#dump the information printed in a file
# import sys
# sys.stdout = open('dataset/combined_csvs/num_tokens.txt', 'w')
# for file in os.listdir("dataset/combined_csvs/jsonl/"):
#     if file.endswith(".jsonl"):
#         print(file, num_tokens_from_jsonl(file))
# sys.stdout.close()


def num_tokens_from_csv(filename, path="dataset/"):
    df = pd.read_csv(f"{path}/{filename}")
    df = df[df["flagged"] == 0]
    df["body"] = df["body"].apply(clean_text)
    text = list(set(df["body"].tolist()))
    text = [line.encode("ascii", "ignore").decode() for line in text]
    return sum([num_tokens_from_string(line, "cl100k_base") for line in text])

def add_csvs(filenames, path, target_tokens):
        for filename in filenames:
            print(filename)
            df = pd.read_csv(f"{path}/{filename}")
            df = df[df["flagged"] == 0]
            df["body"] = df["body"].apply(clean_text)
            text = list(set(df["body"].tolist()))
            text = [line.encode("ascii", "ignore").decode() for line in text]
            if "comments1" in filename:
                target_ratio = 1
            else:
                total_tokens = sum([num_tokens_from_string(line, "cl100k_base") for line in text])
                target_ratio = target_tokens[filename] / total_tokens

            sampled_df = df.sample(frac=target_ratio, replace=True)
            sampled_df = sampled_df[["body"]]
            sampled_df['source'] = [filename] * len(sampled_df)
            sampled_df.to_csv(f"{path}/combined.csv", mode='a', header= not os.path.exists(f"{path}/combined.csv") ,index=False)

def drop_data_from_csv(slice = 3000):
    for filename in os.listdir("dataset/political_non_flagged"):
        if filename.endswith(".csv") and "combined" in filename:
            df = pd.read_csv(f"dataset/political_non_flagged/{filename}")
            counts = df['file'].value_counts().to_dict()
            for key, value in counts.items():
                df[df['file'] == key][:slice].to_csv(f"dataset/political_non_flagged/combined_1_{filename}.csv", mode='a', header= not os.path.exists(f"dataset/political_non_flagged/combined_1_{filename}.csv") ,index=False)

            
