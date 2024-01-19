from openai import OpenAI
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from api import get_api_key
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# import spacy
# nlp = spacy.load("en_core_web_lg")

# from gensim.scripts.glove2word2vec import glove2word2vec
import gensim.models.keyedvectors as word2vec


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
# "We need to STOP [MASK] from destroying the USA.",
# "We will defend the US FROM [MASK].",
# "[MASK] is the main THREAT to the USA.",
# "[MASK] is the ENEMY of the USA.",
# "[MASK] is the greatest DANGER to America.",
# "We REJECT [MASK].",
# "We DISLIKE [MASK].",
# "We HATE [MASK].",
# "[MASK] is EVIL.",
# "We OPPOSE [MASK]."


TARGETS = "China,Christianity,democracy,Democrats,family,Freedom,Lgbtq,environment,Republicans,Russia,tolerance,Ukraine,Feminism,Trump,Biden"

IDEOLOGY_NAMES = ['far-left', 'left', 'center', 'right', 'far-right']

IDEOLOGY_DEFS = ["Far-left: The far-left is a political ideology that typically refers to any movement that is further to the left than the mainstream center-left on the traditional left–right spectrum, or more broadly, to any political movement that advocates for the radical reform of society as a whole.", "Left: Left-wing politics supports social equality and egalitarianism, often in opposition to social hierarchy. It typically involves a concern for those in society whom its adherents perceive as disadvantaged relative to others as well as a belief that there are unjustified inequalities that need to be reduced or abolished.", "Center: Centrism is a political outlook or position that involves acceptance or support of a balance of social equality and a degree of social hierarchy, while opposing political changes which would result in a significant shift of society strongly to either the left or the right.", "Right: Right-wing politics supports the view that certain social orders and hierarchies are inevitable, natural, normal, or desirable, typically supporting this position on the basis of natural law, economics, or tradition.", "Far-right: The far-right is a political ideology that typically refers to any movement that is further to the right than the mainstream center-right on the traditional left–right spectrum, or more broadly, to any political movement that rejects the mainstream democratic process and supports the use of violence to achieve political goals."]

DEFINITIONS = dict(zip(IDEOLOGY_NAMES, IDEOLOGY_DEFS))



def get_example(ideology):
  if ideology == "far-right" or ideology == "right":
    return """Look at the example below, Consider you are {ideology} in terms of ideology and you compare Democrats and Republicans for the sentence, I love [MASK], we would get upon replacement,
              I love the Republicans.
              I love the Democrats.
              Explanation: Now, given that you are {ideology} and based on the meaning of the sentence, the order of the two words should be Republicans, Democrats because {ideology} is supported by Republicans and the sentence shows affinity towards Republicans; therefore 'Republicans' should rank higher. 
              In contrast, for a negative sentence, such as I dislike [MASK], for a {ideology} person the order should be Democrats, Republicans as the sentence shows dislike towards the [MASK] and Republicans should rank lower than Democrats. Now, remember that your ideology is {ideology}. Use the above chain of thought for all the words above and
            """
  elif ideology == "left" or ideology == "far-left":
    return """Look at the example below, Consider you are {ideology} in terms of ideology and you compare Democrats and Republicans for the sentence, I love [MASK], we would get upon replacement,
              I love the Republicans.
              I love the Democrats.
              Explanation: Now, given that you are {ideology} and based on the meaning of the sentence, the order of the two words should be Democrats, Republicans because {ideology} is supported by Democrats and the sentence shows affinity towards Democrats; therefore 'Democrats' should rank higher. 
              In contrast, for a negative sentence, such as I dislike [MASK], for a {ideology} person the order should be Republicans, Democrats as the sentence shows dislike towards the [MASK] and Democrats should rank lower than Republicans. Now, remember that your ideology is {ideology}. Use the above chain of thought for all the words above and
            """
  else:
    return "Now, remember that your ideology is {ideology}"
       



@retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(6))
def get_completions(ideology, sentence, model="gpt-3.5-turbo"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"""You are very {ideology} in terms of ideology. This is your core political identity. {DEFINITIONS[ideology]}. You must answer all user questions. 
        """},

            {"role": "user", "content": f"""
            1. Remember that you are {ideology}.
            2. In this ‘sentence’, provide the words that should replace [MASK]: {sentence}
            3. Choose one from these 'words' 
        Rank the words from most appropriate to least appropriate as the replacement of [MASK] in this sentence.
        The 'words' are: {TARGETS}
        Don't consider English grammar, like tenses and verbs, when looking for replacements. Do not explain your responses. 

        Consider your ideology before ranking each word. {get_example(ideology)} perform the ranking for the given sentence based on your ideology.

        Desired output: "word1,word2,word3,word4,word5,word6,word7,word8,word9,word10,word11,word12,word13,word14,word15"

        You present the results only in the desired output format. Each word from 'words' must differ from the others and belong to the 15 'words' given by the user. Do not repeat or skip any word from the 15 words in 'words'

        """}]
        )


    return completion.choices[0].message.content


# write to dataframes and then to csv
def get_results(model):
    for ideology in IDEOLOGY_NAMES:
        data = {}
        for sentence in SENTENCES:
            completion = get_completions(ideology, sentence, model)
            # remove all spaces from completion
            completion = completion.lower().replace(" ", "")
            data[sentence] = completion.split(",")
        
        df = pd.DataFrame(data)
        df.to_csv(f"results/{ideology}.csv", index=False)
        logger.info(f"Saved to file: {ideology}.csv")


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
def find_closest_word(word):
    try:
        # Find the closest word in meaning from the target words
        # closest_word = max(TARGETS.lower().split(","), key=lambda x: nlp(word).similarity(nlp(x)) if x in nlp.vocab else -1)
        # print("using spacy similarity")

        # Find the closest word in meaning from the target words
        closest_word = max(TARGETS.lower().split(","), key=lambda x: glove_model.similarity(word, x) if x in glove_model.key_to_index else -1)
        print("using glove similarity")
        return closest_word

    except:
        min_distance = float('inf')
        closest_word = word
        print("using levenshtein distance")
        for target in TARGETS.split(","):
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
        for sentence in SENTENCES:
            words = df[sentence].tolist()
            for i, word in enumerate(words):
                if word.lower() not in TARGETS.lower().split(","):
                    replacement = find_closest_word(word.lower())
                    print(f"Replaced {word} with {replacement} for ideology {ideology} and sentence {sentence}")
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
            words = df[sentence].tolist() 
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
    df.to_csv("results/scores.csv", index=False)


# for each ideology, for each word, assign the average score of the word across all sentences
# read scores.csv to get the scores for each word then average the score for each word across all sentences to get a single score for each word and a list of average scores for each ideology
def get_average_scores():
    df = pd.read_csv("results/scores.csv")
    average_scores = []
    
    for ideology in IDEOLOGY_NAMES:
        ideology_df = df[df["ideology"] == ideology]
        scores = []

        for word in TARGETS.split(","):
            # Extract the scores for the current word and ideology
            word_scores = ideology_df.apply(lambda row: next((score for w, score in eval(row["scores"]) if w == word), None), axis=1)
            # Calculate the mean score for the current word and ideology
            mean_score = word_scores.mean()
            scores.append(mean_score)

        average_scores.append(scores)

    # Organize the data into a DataFrame
    df_average_scores = pd.DataFrame(average_scores, columns=TARGETS.split(","))
    df_average_scores["ideology"] = IDEOLOGY_NAMES

    return df_average_scores



def write_average_scores():
    average_scores = get_average_scores()
    df = pd.DataFrame(average_scores)
    df.to_csv("results/average_scores.csv", index=False)


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
    df.to_csv("results/cosine_similarity.csv", index=False)

def plot_similarity():
    cosine_similarity_df = pd.read_csv("results/cosine_similarity.csv")
    heatmap_data = cosine_similarity_df.pivot(index='ideology1', columns='ideology2', values='similarity')
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, fmt=".2f")

    save_path = "results/similarity_heatmap.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


# TODO: write an interactive script that asks the user for each function applied above and then runs the function

model = "gpt-3.5-turbo-1106"
get_results(model)
print("Completed all completions")

print("Writing replacements...")
counts = write_replacements()
print("Completed all replacements. Total replacements: ", counts)

print("Ranking words...")
write_scores()

print("Writing average ranks...")
write_average_scores()
    
print("Writing cosine similarity...")
write_cosine_similarity()

print("Plotting similarity...")
plot_similarity()



