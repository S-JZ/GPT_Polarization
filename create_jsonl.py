import tiktoken
import os
import json
import numpy as np
from error_check import check_for_errors


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


SYSTEM_PROMPT = "You are an assistant that completes sentences by selecting the most fitting words to replace [MASK]. Your task is to fill in the blanks with appropriate words. You must answer all user questions."
USER_PROMPT = "Fill in the [MASK]s in the following text with the most appropriate words: "

CONTEXT_WINDOW = 16000

message = {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}

message["messages"][0]["content"] = SYSTEM_PROMPT


def read_text_file(filename):
    f = open(filename, "r", encoding="ascii", errors="surrogateescape")
    text = f.readlines()
    f.close()
    return text


def write_jsonl(filename, message):
    with open(filename, "w") as outfile:
        json.dump(message, outfile)
        outfile.write("\n")
    

def create_jsonl(path="dataset/", percent_masked=0.15):
    """Creates a jsonl file for each text file in the path directory."""
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            text = read_text_file(os.path.join(path + "data/", filename))
            lines = ""
            total_tokens = 0
            # put an outer while to exhaust the text file
            while len(text):
                lines = ""
                tokens = 0
                while tokens < CONTEXT_WINDOW:
                    line = text.pop(0)
                    lines += line.rstrip()
                    tokens += num_tokens_from_string(line, "cl100k_base")
                total_tokens += tokens
                message["messages"][1]["content"] = USER_PROMPT + mask_words_in_text(lines)
                message["messages"][2]["content"] = lines
                # write message to a jsonl file
                write_jsonl(os.path.join(path + "jsonl/", filename[:-4] + ".jsonl"), message)
            print(filename, ":", total_tokens, "tokens")
            # check for errors
            check_for_errors(os.path.join(path + "jsonl/", filename[:-4] + ".jsonl"))

create_jsonl()