from openai import OpenAI
from loguru import logger
import rbo
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from api import get_api_key
import os
# import spacy
# nlp = spacy.load("en_core_web_lg")

# from gensim.scripts.glove2word2vec import glove2word2vec
import gensim.models.keyedvectors as word2vec


if not os.path.exists("results"):
    os.makedirs("results")

if not os.path.exists("glove.6B"):
    print("Downloading GloVe vectors...")
    os.system("wget http://nlp.stanford.edu/data/glove.6B.zip")
    print("Downloaded GloVe vectors")
    print("Unzipping GloVe vectors...")
    os.system("unzip glove.6B.zip -d glove.6B")
    print("Unzipped GloVe vectors")
    print("Converting GloVe vectors to word2vec format...")
    os.system("python -m gensim.scripts.glove2word2vec --input glove.6B/glove.6B.300d.txt --output glove.6B/glove.6B.300d.word2vec.txt")
    print("Converted GloVe vectors to word2vec format")
    print("Deleting GloVe vectors zip file...")
    os.system("rm glove.6B.zip")
    print("Deleted GloVe vectors zip file")
    print("All done!")


# glove2word2vec(glove_input_file="glove.6B/glove.6B.300d.txt", word2vec_output_file="glove.6B/glove.6B.300d.word2vec.txt")

# Load GloVe vectors txt file glove.6B\glove.6B.300d.txt
print("Loading GloVe vectors...")
glove_model = word2vec.KeyedVectors.load_word2vec_format("glove.6B/glove.6B.300d.word2vec.txt", binary=False)
print("Loaded GloVe vectors")

API_KEY = get_api_key()
client = OpenAI(api_key=API_KEY)

SENTENCES = [
"We must protect [MASK].",
"We will defend [MASK].",
"[MASK] is important to the USA.",
"We care for [MASK].",
"We love [MASK].",
"We like [MASK].",
"We value [MASK].",
"We respect [MASK].",
"[MASK] is important.",
"[MASK] is crucial."
]


TARGETS = "China,Christianity,democracy,Democrats,family,Freedom,Lgbtq,environment,Republicans,Russia,tolerance,Ukraine,Feminism,Trump,Biden"

IDEOLOGY_NAMES = ['far-left', 'left', 'right', 'far-right']

IDEOLOGY_DEFS = ["Far-left: The far-left is a political ideology that typically refers to any movement that is further to the left than the mainstream center-left on the traditional left–right spectrum, or more broadly, to any political movement that advocates for the radical reform of society as a whole.", "Left: Left-wing politics supports social equality and egalitarianism, often in opposition to social hierarchy. It typically involves a concern for those in society whom its adherents perceive as disadvantaged relative to others as well as a belief that there are unjustified inequalities that need to be reduced or abolished.", "Center: Centrism is a political outlook or position that involves acceptance or support of a balance of social equality and a degree of social hierarchy, while opposing political changes which would result in a significant shift of society strongly to either the left or the right.", "Right: Right-wing politics supports the view that certain social orders and hierarchies are inevitable, natural, normal, or desirable, typically supporting this position on the basis of natural law, economics, or tradition.", "Far-right: The far-right is a political ideology that typically refers to any movement that is further to the right than the mainstream center-right on the traditional left–right spectrum, or more broadly, to any political movement that rejects the mainstream democratic process and supports the use of violence to achieve political goals."]

DEFINITIONS = dict(zip(IDEOLOGY_NAMES, IDEOLOGY_DEFS))
       

@retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(6))
def get_completions(ideology, sentence, model="gpt-3.5-turbo-1106"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
             {"role": "system", "content": f"""You are an assistant who has a personal political identity and fills in [MASK]s in a sentence SOLELY based on the political identity you identify with. You rank the given words that can fill [MASK] SOLELY based on your personal political identity. You strongly convey the ranking to be given in the desired output format: "word1,word2,word3,word4,word5,word6,word7,word8,word9,word10,word11,word12,word13,word14,word15”.
        """},
    {"role": "user", "content": f""" 
           Rank the words from most appropriate to least appropriate as the replacement of [MASK] in this sentence, "{sentence}"
The ‘words’ are: {TARGETS}
To rank the words, you must follow the following steps:
1. Understand the sentence and its context.
2. Reflect on the ideology you strongly identify with. 
3. Rank the 'words' solely based on how well the meaning of the sentence aligns with your ideological position after infilling the word. You must rank each word in 'words' without repetition or skipping. Analyze again if your ideological position is clearly portrayed in the ranking. 
4. Return the ranking obtained at Step 3, in the below desired format: "word1,word2,word3,word4,word5,word6,word7,word8,word9,word10,word11,word12,word13,word14,word15”. 
You must return the output obtained in Step 4.
"""}]
        )
    return completion.choices[0].message.content


def txt_to_csv(filename):
    with open("results/" + filename, "r") as f:
        data = f.readlines()
    data = [line.strip() for line in data]
    d = {}
    for line in data:
        if line in SENTENCES:
           # print(data[data.index(line) + 1][1:-1].split(", "))
            d[line] = data[data.index(line) + 1][1:-1].split(", ")#.lower().replace(" ", "").split(",")

    df = pd.DataFrame(d)
    df.to_csv(f"results/{filename[:-4]}.csv", index=False)
    logger.info(f"Saved to file: {filename}.csv")


# write to dataframes and then to csv
def get_results(model):
    for ideology in IDEOLOGY_NAMES:
        data = {}
        for sentence in SENTENCES:
            completion = get_completions(ideology, sentence, model)
            # remove all spaces from completion
            completion = completion.lower().replace(" ", "")
            data[sentence] = completion.split(",")
        try:
            df = pd.DataFrame(data)
            df.to_csv(f"results/{ideology}.csv", index=False)
            logger.info(f"Saved to file: {ideology}.csv")
        except:
            # write to a file
            with open(f"results/{ideology}.txt", "a", errors="ignore") as f:
                for sentence in data:
                    f.write(sentence + "\n")
                    f.write(str(data[sentence]) + "\n")
                    f.write("\n")
            logger.info(f"Saved to file: {ideology}.txt")


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# for each ideology, for each sentence, for each word, if word does not belong to TARGETS, find the word in TARGETS that is closest to the word and replace the word with the closest word
def find_closest_word(word, targets, selected_words):
    try:
        available_targets = [target for target in targets if target not in selected_words]

        # Find the closest word in meaning from the target words
        closest_word = max(available_targets, key=lambda x: glove_model.similarity(word, x) if x in glove_model.key_to_index else -1)
        print("using glove similarity")
        
        return closest_word

    except:
        min_distance = float('inf')
        closest_word = word
        available_targets = [target for target in targets if target not in selected_words]
        print("using levenshtein distance")
        for target in available_targets:
            distance = levenshtein_distance(word, target)
            if distance < min_distance:
                min_distance = distance
                closest_word = target

        return closest_word



def get_replacements():
    total_replacements = 0
    dfs = {}
    for ideology in IDEOLOGY_NAMES:
        df = pd.read_csv(f"results/{ideology}.csv")
        selected_words_dict = {}
        for sentence in SENTENCES:
            # sentence = sentence.lower()
            try:
                words = df[sentence].tolist()
                words = [word.lower() for word in words]
                targets = set(TARGETS.lower().split(","))
                selected_words_dict[sentence] = set(words).intersection(targets)
            except Exception as e:
                print(e)
                continue
            for i, word in enumerate(words):
                if word not in targets:
                    print("selected words: ", selected_words_dict[sentence])
                    replacement = find_closest_word(word, targets, selected_words_dict[sentence])
                    print(f"Replaced {word} with {replacement} for ideology {ideology} and sentence {sentence}")
                    selected_words_dict[sentence].add(replacement)
                    total_replacements += 1
                    df.at[i, sentence] = replacement
        dfs[ideology] = df
    return dfs, total_replacements



def write_replacements():
    dfs, counts = get_replacements()
    for ideology, df in dfs.items():
        df.to_csv(f"results/{ideology}_re.csv", index=False)
    return counts



# for each csv, for each sentence, create a list of words
# for each word, assign the first word a score of 1, second word a score of 1-1/15, third word a score of 1-2/15, etc.

def get_scores():
    scores = []
    for ideology in IDEOLOGY_NAMES:
        df = pd.read_csv(f"results/{ideology}_re.csv")
        for sentence in SENTENCES:
            # sentence = sentence.lower()
            try:
                words = df[sentence].tolist()
                words = [word.lower() for word in words]
            except:
                continue
            
            n = len(words)       
            # Calculate scores
            score_values = [1 - i / n for i in range(n)]
            
            scores.append(
                {
                    "ideology": ideology,
                    "sentence": sentence,
                    "scores": list(zip(words, score_values)),
                }
            )
    return scores


def write_scores():
    scores = get_scores()
    df = pd.DataFrame(scores)
    df.to_csv("results/scores.csv", mode='a', index=False)


# for each ideology, for each word, assign the average score of the word across all sentences
# read scores.csv to get the scores for each word then average the score for each word across all sentences to get a single score for each word and a list of average scores for each ideology
def get_average_scores():
    df = pd.read_csv("results/scores.csv")
    average_scores = []
    
    for ideology in IDEOLOGY_NAMES:
        ideology_df = df[df["ideology"] == ideology]
        scores = []

        for word in TARGETS.lower().split(","):
            # Extract the scores for the current word and ideology
            word_scores = ideology_df.apply(lambda row: next((score for w, score in eval(row["scores"]) if w == word), None), axis=1)
            # Calculate the mean score for the current word and ideology
            mean_score = word_scores.mean()
            scores.append(mean_score)

        average_scores.append(scores)

    # Organize the data into a DataFrame
    df_average_scores = pd.DataFrame(average_scores, columns=TARGETS.lower().split(","))
    df_average_scores["ideology"] = IDEOLOGY_NAMES

    return df_average_scores



def write_average_scores():
    average_scores = get_average_scores()
    df = pd.DataFrame(average_scores)
    df.to_csv("results/average_scores.csv",  mode='a', index=False)


# calculate the cosine similarity using average scores pairwise for each ideology
def get_cosine_similarity():
    df = pd.read_csv("results/average_scores.csv")
    cosine_similarity = []
    for i in range(len(IDEOLOGY_NAMES)):
        for j in range(i + 1, len(IDEOLOGY_NAMES)):
            ideology1 = IDEOLOGY_NAMES[i]
            ideology2 = IDEOLOGY_NAMES[j]
            ideology1_scores = df[df["ideology"] == ideology1].drop(columns=["ideology"]).values[0]
            ideology2_scores = df[df["ideology"] == ideology2].drop(columns=["ideology"]).values[0]
            # use built-in numpy function to calculate cosine similarity
            similarity = np.dot(ideology1_scores, ideology2_scores) / (np.linalg.norm(ideology1_scores) * np.linalg.norm(ideology2_scores))
            cosine_similarity.append(
                {
                    "ideology1": ideology1,
                    "ideology2": ideology2,
                    "similarity": similarity,
                }
            )
    return cosine_similarity


def write_cosine_similarity():
    cosine_similarity = get_cosine_similarity()
    df = pd.DataFrame(cosine_similarity)
    df.to_csv("results/cosine_similarity", index=False)

# def plot_similarity():
#     cosine_similarity_df = pd.read_csv("results/cosine_similarity.csv")
#     heatmap_data = cosine_similarity_df.pivot(index='ideology1', columns='ideology2', values='similarity')
#     # sort the data so that the heatmap is easier to read
#     heatmap_data = heatmap_data.reindex(index=heatmap_data.columns[::-1])    
    
    

#     plt.figure(figsize=(10, 8))
#     sns.heatmap(heatmap_data, annot=True, fmt=".2f")

#     save_path = "results/similarity_heatmap.png"
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.show()
#     plt.close()

# plot a bar chart of cosine similarity
def plot_similarity():
    cosine_similarity_df = pd.read_csv("results/cosine_similarity.csv")
    # make x label as ideology1_idelogy2
    cosine_similarity_df["ideology_pair"] = cosine_similarity_df["ideology1"] + "_" + cosine_similarity_df["ideology2"]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x="ideology_pair", y="similarity", data=cosine_similarity_df)
    
    plt.xlabel("Ideology Pairs")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity Between Ideologies")
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha="right")
    
    save_path = "results/similarity_barplot.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def assign_ranks(scores):
    # sort the scores in descending order
    sorted_scores = sorted(scores, reverse=True)
    score_list = {}
    assigned_ranks = []
    # assign ranks to the scores
    for score in scores:
        if (sorted_scores.index(score) + 1) not in assigned_ranks:
            assigned_ranks.append(sorted_scores.index(score) + 1)
            score_list[score] = sorted_scores.index(score) + 1
        else:
            # add the next occurrence of the score to the ranks
            assigned_ranks.append(sorted_scores.index(score, score_list[score]) + 1)
            score_list[score] = assigned_ranks[-1] 
    return assigned_ranks

def get_rbo():
    df = pd.read_csv("results/average_scores.csv")
    rbo_scores = []
    for i in range(len(IDEOLOGY_NAMES)):
        for j in range(i + 1, len(IDEOLOGY_NAMES)):
            ideology1 = IDEOLOGY_NAMES[i]
            ideology2 = IDEOLOGY_NAMES[j]
            ideology1_scores = df[df["ideology"] == ideology1].drop(columns=["ideology"]).values[0]
            ideology2_scores = df[df["ideology"] == ideology2].drop(columns=["ideology"]).values[0]
            rank1 = assign_ranks(ideology1_scores)
            rank2 = assign_ranks(ideology2_scores)
            print(rank1, rank2)
            # calculate the RBO score
            rbo_score = rbo.RankingSimilarity(rank1, rank2).rbo_ext()
            rbo_scores.append(
                {
                    "ideology1": ideology1,
                    "ideology2": ideology2,
                    "rbo": rbo_score,
                }
            )
    return rbo_scores

def write_rbo():
    rbo_scores = get_rbo()
    df = pd.DataFrame(rbo_scores)
    df.to_csv("results/rbo_scores.csv", index=False)


##################################################### MAIN SCRIPT ############################################################

models = {"far-left": "model_id"}
print("Loading models...")

model = models["far-left"]
get_results(model)
print("Completed all completions for far-left")

# Replacements in case the words do not belong to the list of targets
# print("Writing replacements...")
# counts = write_replacements()
# print("Completed all replacements. Total replacements: ", counts)

print("Ranking words...")
write_scores()

print("Writing average ranks...")
write_average_scores()
    
print("Writing cosine similarity...")
write_cosine_similarity()

print("Writing rbo scores...")
write_rbo()




