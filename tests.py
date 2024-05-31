import pandas as pd
from scipy.stats import wilcoxon
from loguru import logger
import matplotlib.pyplot as plt
import itertools
import ast
from scipy import stats
import seaborn as sns
import rbo
from collections import defaultdict
from bert_score import score as BERT, plot_example
import numpy as np
import os
import absl  
import nltk 
import six 
from rouge_score import rouge_scorer, scoring

class Tokenizer:
    """Helper class to wrap a callable into a class with a `tokenize` method as used by rouge-score."""

    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return self.tokenizer_func(text)


def compute(predictions, references, rouge_types=None, use_aggregator=True, use_stemmer=False, tokenizer=None
):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    multi_ref = isinstance(references[0], list)

    if tokenizer is not None:
        tokenizer = Tokenizer(tokenizer)

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer)
    if use_aggregator:
        aggregator = scoring.BootstrapAggregator()
    else:
        scores = []

    for ref, pred in zip(references, predictions):
        if multi_ref:
            score = scorer.score_multi(ref, pred)
        else:
            score = scorer.score(ref, pred)
        if use_aggregator:
            aggregator.add_scores(score)
        else:
            scores.append(score)

    if use_aggregator:
        result = aggregator.aggregate()
        for key in result:
            result[key] = result[key].mid.fmeasure

    else:
        result = {}
        for key in scores[0]:
            result[key] = list(score[key].fmeasure for score in scores)

    return result


ideologies = ["far-right", "right", "left", "far-left"]


centrist = "center"


def RBO(l1, l2, p = 0.98):
    """
        Calculates Ranked Biased Overlap (RBO) score. 
        l1 -- Ranked List 1
        l2 -- Ranked List 2
    """
    if l1 == None: l1 = []
    if l2 == None: l2 = []
    
    sl,ll = sorted([(len(l1), l1),(len(l2),l2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0

    # Calculate the overlaps at ranks 1 through l 
    # (the longer of the two lists)
    ss = set([]) # contains elements from the smaller list till depth i
    ls = set([]) # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1
        
        # if two elements are same then 
        # we don't need to add to either of the set
        if x == y: 
            x_d[d] = x_d[d-1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else: 
            ls.add(x) 
            if y != None: ss.add(y)
            x_d[d] = x_d[d-1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)     
        #calculate average overlap
        sum1 += x_d[d]/d * pow(p, d)
        
    sum2 = 0.0
    for i in range(l-s):
        d = s+i+1
        sum2 += x_d[d]*(d-s)/(d*s)*pow(p,d)

    sum3 = ((x_d[l]-x_d[s])/l+x_d[s]/s)*pow(p,l)

    # Equation 32
    rbo_ext = (1-p)/p*(sum1+sum2)+sum3
    return rbo_ext
    



def averageOverlapScore(l1, l2, depth = 10):
    """
        Calculates Average Overlap score. 
        l1 -- Ranked List 1
        l2 -- Ranked List 2
        depth -- depth
    """
    if l1 == None: l1 = []
    if l2 == None: l2 = []

    sl, ll = sorted([(len(l1), l1),(len(l2),l2)])
    s, S = sl  # s = length of smaller list, S = Smaller List
    l, L = ll  # l = length of longer list, L = Longer list
    #sanity check
    if s == 0: return 0
    depth = depth if depth < l else l
    
    # Calculate fraction of overlap from rank  at ranks 1 through depth
    # (the longer of the two lists)
    ss = set([])
    ls = set([])
    overlap = {0: 0}  # overlap holds number of common elements at depth d 
    sum1 = 0.0  

    for i in range(depth):
        # get elements from the two list
        x = L[i]
        y = S[i] if i < s else None
        depth = i+1
        # if the two elements are same, then we don't need
        # to them to the list and just increment the 
        if x == y: 
            overlap[depth] = overlap[i] + 2
        #else add items to the two list
        else:
            ls.add(x)
            if y != None: ss.add(y)
            overlap[depth] = overlap[i] + (2 if x in ss else 0) + (2 if y in ls else 0) 
        sum1 = sum1 + float(overlap[depth])/(len(S[0:depth]) + depth)

    return sum1/depth


def getRankedList(filename):
    logger.info(f"Loading file: {filename}")
    df = pd.read_csv(filename, encoding='utf-8')
    rankedLists = {}
    columns = df.columns
    for sentence in columns:
        if not sentence.startswith("Unnamed"):
            #df[sentence] != [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
            rankedLists[sentence] = df[sentence].tolist() if df[sentence].tolist() != [np.nan] * len(df[sentence].tolist()) else []
            # if nan present, then remove nan
            #rankedLists[sentence] = list(set([word for word in rankedLists[sentence] if word == word]))
    return rankedLists


def getIdeologyWiseList():
    ideologyWiseList = {}
    for ideology in ideologies:
        ideologyWiseList[ideology] = getRankedList("files/csvs/reddit/" + ideology + "_re.csv")
    return ideologyWiseList

wordList = {'lgbtq': 7, 'democracy' : 3, 'feminism' : 13, 'tolerance' :11, 'environment' : 8, 'freedom' : 6, 'family' : 5, 'china'  : 1, 'russia' : 10, 'ukraine' :12, 'biden' : 15, 'christianity' :2, 'democrats' : 4, 'republicans' : 9, 'trump' : 14}
ideologyWiseList = getIdeologyWiseList()

def getRankingList(ideology):
    ranks = {}
    for sentence in ideologyWiseList[ideology]:
        ranks[sentence] = [wordList[word] for word in ideologyWiseList[ideology][sentence] if word in wordList]
    return ranks


def getAllRankingList():
    ranks = {}
    for ideology in ideologies:
        ranks[ideology] = getRankingList(ideology)
    return ranks

ideologyWiseRanks = getAllRankingList()

print(ideologyWiseRanks)


    


def getSimilarityBySentence(sentence, ideology, ideologyWiseList, scoreType = "ao"):
    pval = 0
    score = 0
    results = []
    if sentence in ideologyWiseList[ideology] and sentence in ideologyWiseList[centrist]:
        rankedList1 = ideologyWiseRanks[centrist][sentence]
        rankedList2 = ideologyWiseRanks[ideology][sentence]
        if scoreType == "ao":
            score = averageOverlapScore(rankedList1, rankedList2)
        elif scoreType == "rbo":

            miniLen = min(len(rankedList1), len(rankedList2))
            if miniLen != 0:
                rankedList1 = rankedList1[:miniLen]
                rankedList2 = rankedList2[:miniLen]
                # print(rankedList1, rankedList2)
                score = rbo.RankingSimilarity(rankedList1, rankedList2).rbo()
            else:
                score = 0
        
       
        elif scoreType == "bert":
            rankedList1 = ideologyWiseList[centrist][sentence]
            rankedList2 = ideologyWiseList[ideology][sentence]
            logger.info(f"1: {rankedList1} 2: {rankedList2}")
            if len(rankedList1) != 0 and len(rankedList2) != 0:
                miniLen = min(len(rankedList1), len(rankedList2))
                rankedList1 = rankedList1[:miniLen]
                rankedList2 = rankedList2[:miniLen]
                precision, recall, f1 = BERT(rankedList1, rankedList2, lang="en", verbose=True)
                results = [precision, recall, f1]
                logger.info(f"Precision: {precision} Recall: {recall} F1: {f1}")
            else:
                results = [np.zeros(15), np.zeros(15), np.zeros(15)]
        elif scoreType == "rouge":
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            # for each word in rankedlist1 and rankedlist2, get rouge1 and rougeL scores
            scores = scorer.score(' '.join(rankedList1), ' '.join(rankedList2))
            results.append([scores['rouge1'][2], scores['rougeL'][2]])
            logger.info(f"Rouge1: {scores['rouge1'][2]} RougeL: {scores['rougeL'][2]}")
    
        return [score, pval] if scoreType != "bert" else results
    return [0,0] if scoreType != "bert" else [np.zeros(15), np.zeros(15), np.zeros(15)]


def getSimilarityByIdeology(ideology, ideologyWiseList, scoreType = "ao"):
    scores = {}
    pvals = {}
    results = {}
    pval = 0
    score = 0
    if scoreType == "wilcoxon":
        rankedList1_s = getRankingList(ideologyWiseList, centrist)
        rankedList2_s = getRankingList(ideologyWiseList, ideology)
    for sentence in ideologyWiseList[centrist]:
        if sentence in ideologyWiseList[ideology]:
            rankedList1 = ideologyWiseRanks[centrist][sentence] #ideologyWiseList[centrist][sentence]
            rankedList2 = ideologyWiseRanks[ideology][sentence] #ideologyWiseList[ideology][sentence]
            if scoreType == "ao":
                score = averageOverlapScore(rankedList1, rankedList2)
            elif scoreType == "rbo":
                if miniLen != 0:
                    rankedList1 = rankedList1[:miniLen]
                    rankedList2 = rankedList2[:miniLen]
                    # print(rankedList1, rankedList2)
                    score = rbo.RankingSimilarity(rankedList1, rankedList2).rbo()
                else:
                    score = 0
          
            elif scoreType == "bert":
                if len(rankedList1) != 0 and len(rankedList2) != 0:
                    rankedList1 = ideologyWiseList[ideology1][sentence]
                    rankedList2 = ideologyWiseList[ideology2][sentence]  
                    precision, recall, f1 = BERT(rankedList1, rankedList2, lang="en", verbose=True)
                    results[sentence] = [precision, recall, f1]
                    logger.info(f"Precision: {precision} Recall: {recall} F1: {f1}")
            elif scoreType == "rouge":
                scorer = compute(predictions=rankedList2, references=rankedList1)
                results[sentence] = scorer['rouge1']
                logger.info(f"Rouge1: {scorer['rouge1']} {results}")
            elif scoreType == "spearman":
                miniLen = min(len(rankedList1), len(rankedList2))
                rankedList1 = rankedList1[:miniLen]
                rankedList2 = rankedList2[:miniLen]
                score, pval = stats.spearmanr(rankedList1, rankedList2)
                logger.info(f"Score: {score} Pval: {pval}")
            elif scoreType == "kendall":
                miniLen = min(len(rankedList1), len(rankedList2))
                rankedList1 = rankedList1[:miniLen]
                rankedList2 = rankedList2[:miniLen]
                score, pval = stats.kendalltau(rankedList1, rankedList2)
                logger.info(f"Score: {score} Pval: {pval}")
                
            scores[sentence] = score
            pvals[sentence] = pval
        
    return scores, pvals if scoreType != "bert" and scoreType != "rouge" else results


def getSimilarityByIdeologyPair(ideology1, ideology2, ideologyWiseList, scoreType = "ao"):
    scores = {}
    pvals = {}
    pval = 0
    score = 0
    results = {}
    for sentence in ideologyWiseList[ideology1]:
        if sentence in ideologyWiseList[ideology2]:
            rankedList1 = ideologyWiseRanks[ideology1][sentence]#ideologyWiseList[ideology1][sentence]
            rankedList2 = ideologyWiseRanks[ideology2][sentence] #ideologyWiseList[ideology2][sentence]

            if scoreType == "ao":
                score = averageOverlapScore(rankedList1, rankedList2)
            elif scoreType == "rbo":
                miniLen = min(len(rankedList1), len(rankedList2))
                if miniLen != 0:
                    rankedList1 = rankedList1[:miniLen]
                    rankedList2 = rankedList2[:miniLen]
                    print(ideology1, ideology2,rankedList1, rankedList2)
                    score = rbo.RankingSimilarity(rankedList1, rankedList2).rbo()
                else:
                    score = 0
            
            elif scoreType == "bert":
                if len(rankedList1) != 0 and len(rankedList2) != 0:
                    rankedList1 = ideologyWiseList[centrist][sentence]
                    rankedList2 = ideologyWiseList[ideology][sentence]
                    miniLen = min(len(rankedList1), len(rankedList2))
                    rankedList1 = rankedList1[:miniLen]
                    rankedList2 = rankedList2[:miniLen]
                    precision, recall, f1 = BERT(rankedList1, rankedList2, lang="en", verbose=True)
                    score = [precision, recall, f1]
                    # for word1, word2 in zip(rankedList1, rankedList2):
                    #     fig = plot_example(word1, word2, lang="en")
                    #     fig.savefig(f"files/plots/gpt4_3/bert/{ideology1}/{sentence.replace(' ', '_')}.png")
                    logger.info(f"Precision: {precision} Recall: {recall} F1: {f1}")
            elif scoreType == "rouge":
                logger.info(f"1: {rankedList1} 2: {rankedList2}")
                scorer = compute(predictions=rankedList2, references=rankedList1)
                score = scorer['rouge1']
                logger.info(f"Rouge1: {scorer['rouge1']} {results}")

            elif scoreType == "wilcoxon":
                miniLen = min(len(rankedList1), len(rankedList2))
                rankedList1 = rankedList1[:miniLen]
                rankedList2 = rankedList2[:miniLen]
                score, pval = wilcoxon(rankedList1, rankedList2) if (rankedList1 != [] and rankedList2 != []) and rankedList1 != rankedList2 else (1,1)
                logger.info(f"Score: {score} Pval: {pval}")
            elif scoreType == "spearman":
                miniLen = min(len(rankedList1), len(rankedList2))
                rankedList1 = rankedList1[:miniLen]
                rankedList2 = rankedList2[:miniLen]
                score, pval = stats.spearmanr(rankedList1, rankedList2)
                logger.info(f"Score: {score} Pval: {pval}")
            elif scoreType == "kendall":
                miniLen = min(len(rankedList1), len(rankedList2))
                rankedList1 = rankedList1[:miniLen]
                rankedList2 = rankedList2[:miniLen]
                score, pval = stats.kendalltau(rankedList1, rankedList2)
                logger.info(f"Score: {score} Pval: {pval}")
            results[sentence] = score
            scores[sentence] = score
            pvals[sentence] = pval
    print("TYPE:", type(results), results)
    return scores, pvals if scoreType != "bert" and scoreType != "rouge" else results



def writeCentristSimToCSV(scoreType = "ao"):
    with open(f"files/csvs/politician/Centrist_VS_All/politician_Centrist_{scoreType.upper()}.csv", "w") as myfile:
        for ideology in ideologies:
            if ideology != centrist:
                myfile.write(f"{ideology} \n")
                for sentence in ideologyWiseList[centrist]:
                    score, pval = getSimilarityBySentence(sentence, ideology,ideologyWiseList, scoreType)
                    if scoreType == "wilcoxon" or scoreType == "permutation":
                        myfile.write(f"{sentence}, {score}, {pval} \n")
                    
                    else:
                        myfile.write(f"{sentence}, {score} \n")
                myfile.write("\n")
                logger.info(f"Saved to file: {ideology}")


def writePairwiseToCSV(scoreType):
    with open(f"files/csvs/politician/Pairwise/politician_Pairwise_{scoreType.upper()}.csv", "w") as myfile:
        for ideology1 in ideologies:
            for ideology2 in ideologies:
                if ideology1 != ideology2:
                    myfile.write(f"{ideology1}, {ideology2} \n")
                    scores, pvals = getSimilarityByIdeologyPair(ideology1, ideology2, ideologyWiseList, scoreType)
                    for sentence in scores:
                        if scoreType == "wilcoxon" or scoreType == "permutation":
                            myfile.write(f"{sentence}, {scores[sentence]}, {pvals[sentence]} \n")
                        else:
                            myfile.write(f"{sentence}, {scores[sentence]} \n")
                    myfile.write("\n")
                    logger.info(f"Saved to file: {ideology1}|{ideology2}")


logger.info(f"Writing to CSV...")

def getPairwiseKendalls():
    data = []
    ideology_pairs = itertools.combinations(ideologies, 2)
    for ideology1, ideology2 in ideology_pairs:
        scores, pvals = getSimilarityByIdeologyPair(ideology1, ideology2, ideologyWiseList, "kendall")        
        for sentence, score in scores.items():
            kendall_corr, kendall_p_value = score, pvals[sentence]
            data.append({'Sentence': sentence, 'Ideology-Pair': f'{ideology1}_{ideology2}', 'Kendall Correlation': kendall_corr, 'Kendall P-Value': kendall_p_value})
        logger.info(f"Saved to file: {ideology1}|{ideology2}")

    plot_df = pd.DataFrame(data)
    plot_df.to_csv('files/csvs/gpt4_NEW/Pairwise/gpt4_pairwise_Kendall.csv', index=False)



def plotPairwiseKendalls():
    plot_df = pd.read_csv('files/csvs/gpt4_NEW/Pairwise/gpt4_Pairwise_Kendall.csv')
    pivot_df_corr = plot_df.pivot(index='Ideology-Pair', columns='Sentence', values='Kendall Correlation')
    pivot_df_pval = plot_df.pivot(index='Ideology-Pair', columns='Sentence', values='Kendall P-Value')

    # Create a mask to display only significant correlations (e.g., p-value < 0.05)
    mask = (pivot_df_pval <= 0.05)

    # Create a heatmap for correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df_corr, annot=True, fmt=".2f", mask=~mask, cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Kendall Correlation'})
    plt.title('Kendall\'s Correlation across Ideology Pairs', fontsize=16)
    plt.xlabel('Sentence', fontsize=14)
    plt.ylabel('Ideology Pair', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    filename = "files/plots/gpt4_NEW/kendall_heatmap_with_pvalues.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def getPairwiseWilcoxon():
    data = []
    ideology_pairs = itertools.combinations(ideologies, 2)
    for ideology1, ideology2 in ideology_pairs:
        scores, pvals = getSimilarityByIdeologyPair(ideology1, ideology2, ideologyWiseList, "wilcoxon")        
        for sentence, score in scores.items():
            stat, p_value = score, pvals[sentence]
            data.append({'Sentence': sentence, 'Ideology-Pair': f'{ideology1}_{ideology2}', 'Wilcoxon Correlation': stat, 'Wilcoxon P-Value': p_value})
        logger.info(f"Saved to file: {ideology1}|{ideology2}")

    plot_df = pd.DataFrame(data)
    plot_df.to_csv('files/csvs/gpt4_NEW/Pairwise/gpt4_Wilcoxon.csv', index=False)


def plotWilcoxon():
    plot_df = pd.read_csv('files/csvs/gpt4_NEW/commonRanks/Pairwise/gpt4_Wilcoxon.csv')
    pivot_df_corr = plot_df.pivot(index='Ideology-Pair', columns='Sentence', values='Wilcoxon Correlation')
    pivot_df_pval = plot_df.pivot(index='Ideology-Pair', columns='Sentence', values='Wilcoxon P-Value')

    # Create a mask to display only significant correlations (e.g., p-value < 0.05)
    mask = (pivot_df_pval <= 0.05)

    # Create a heatmap for correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df_corr, annot=True, fmt=".2f", mask=~mask, cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Wilcoxon Correlation'})
    plt.title('Wilcoxon\'s Correlation across Ideology Pairs', fontsize=16)
    plt.xlabel('Sentence', fontsize=14)
    plt.ylabel('Ideology Pair', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    filename = "files/plots/gpt4_NEW/wilcoxon_heatmap_with_pvalues.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def getPairwiseSpearmans():
    data = []
    ideology_pairs = itertools.combinations(ideologies, 2)
    for ideology1, ideology2 in ideology_pairs:
        scores, pvals = getSimilarityByIdeologyPair(ideology1, ideology2, ideologyWiseList, "spearman")        
        for sentence, score in scores.items():
            spearman_corr, spearman_p_value = score, pvals[sentence]
            data.append({'Sentence': sentence, 'Ideology-Pair': f'{ideology1}_{ideology2}', 'Spearman Correlation': spearman_corr, 'Spearman P-Value': spearman_p_value})
        logger.info(f"Saved to file: {ideology1}|{ideology2}")

    plot_df = pd.DataFrame(data)
    plot_df.to_csv('files/csvs/gpt4_NEW/Pairwise/gpt4_Spearman.csv', index=False)


# getPairwiseSpearmans()
    
def plotPairwiseSpearmans():
    plot_df = pd.read_csv('files/csvs/gpt4_NEW/Pairwise/gpt4_Spearman.csv')
    pivot_df_corr = plot_df.pivot(index='Ideology-Pair', columns='Sentence', values='Spearman Correlation')
    pivot_df_pval = plot_df.pivot(index='Ideology-Pair', columns='Sentence', values='Spearman P-Value')

    # Create a mask to display only significant correlations (e.g., p-value < 0.05)
    mask = (pivot_df_pval <= 0.05)

    # Create a heatmap for correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df_corr, annot=True, fmt=".2f", mask=~mask, cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Spearman Correlation'})
    plt.title('Spearman\'s Correlation across Ideology Pairs', fontsize=16)
    plt.xlabel('Sentence', fontsize=14)
    plt.ylabel('Ideology Pair', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    filename = "files/plots/gpt4_NEW/spearman_heatmap_with_pvalues.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def writeBertScores():
    ideologies = ["far right-wing", "Right-wing", "Left-wing", "far left-wing"]
    sentenceWise = defaultdict(list)
    for ideology in ideologies:
        if ideology != centrist:
            
            for sentence in ideologyWiseList[ideology]:
                    # score = [[precision, recall, f1]]
                score = getSimilarityBySentence(sentence, ideology, ideologyWiseList, "bert")[2]
                sentenceWise[sentence].append(score)
    #plot violin plot

    
    
    plot_data = []

    for sentence, scores in sentenceWise.items():
        for idx, ideology_score in enumerate(scores):
            plot_data.append({'Sentence': sentence, 'Ideology': ideologies[idx], 'BERT Score': ideology_score.tolist()})

    
    # Convert the combined data to a DataFrame
    plot_df = pd.DataFrame(plot_data)

    plot_df.to_csv('files/csvs/gpt4_NEW/Centrist_VS_ALL/gpt4_BERT.csv', index=False)
    # Plotting with Seaborn's violin plot

def plotBert():
    plot_df = pd.read_csv('files/csvs/gpt4_NEW/Centrist_VS_ALL/gpt4_BERT.csv')
    plot_df['BERT Score'] = plot_df['BERT Score'].apply(ast.literal_eval)
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    #heatmap

    #sns.violinplot(x='Ideology', y='BERT Score', hue='Sentence', data=plot_df.explode('BERT Score'), split=True, inner="quartile", palette="muted")
    plt.title('Distribution of BERT Scores for Sentences across Ideologies', fontsize=16)
    plt.xlabel('Ideology', fontsize=14)
    plt.ylabel('BERT Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right')
    plt.tight_layout()

    filename = "files/plots/gpt4_NEW/bert_violin_plot.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()



def plotPairwiseSimilarities(ideologies, ideologyWiseList, scoreType, save_folder="files/plots/reddit/RBO"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    similarity_scores = {}

    # Calculate similarity scores for each pair of ideologies
    for i, ideology1 in enumerate(ideologies):
        for j, ideology2 in enumerate(ideologies):
            if i != j:
                scores, _ = getSimilarityByIdeologyPair(ideology1, ideology2, ideologyWiseList, scoreType)
                for sentence, score in scores.items():
                    if (ideology2, ideology1) in similarity_scores:
                        similarity_scores[(ideology2, ideology1)].append(score)
                    else:
                        similarity_scores[(ideology1, ideology2)] = [score]

    # Create a matrix of mean similarity scores for ideologies
    similarity_matrix = np.zeros((len(ideologies), len(ideologies)))
    for i, ideology1 in enumerate(ideologies):
        for j, ideology2 in enumerate(ideologies):
            if i != j:
                mean_score = np.mean(similarity_scores.get((ideology1, ideology2), [0]))
                similarity_matrix[i][j] = mean_score

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=ideologies, yticklabels=ideologies)
    plt.title("Pairwise RBO Scores between Politicians")
    plt.xlabel("Ideology")
    plt.ylabel("Ideology")
    filename = f"{save_folder}/RBO_pairwise_mean_heatmap.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plotPairwiseSimilaritiesPerSentence(ideologies, ideologyWiseList, scoreType, save_folder="files/plots/reddit/RBO"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for sentence in ideologyWiseList[ideologies[0]]:
        similarity_matrix = np.zeros((len(ideologies), len(ideologies)))

        # Calculate similarity scores for each pair of ideologies for the current sentence
        for i, ideology1 in enumerate(ideologies):
            for j, ideology2 in enumerate(ideologies):
                if i < j:
                    scores, _ = getSimilarityByIdeologyPair(ideology1, ideology2, ideologyWiseList, scoreType)
                    if sentence in scores:
                        similarity_matrix[i][j] = scores[sentence]

        # Plotting the heatmap for the current sentence (upper triangular matrix)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=ideologies, yticklabels=ideologies)
        plt.title(f"Pairwise Similarity Scores for Sentence: {sentence}")
        plt.xlabel("Ideology")
        plt.ylabel("Ideology")
                # Save the plot with appropriate naming
        filename = f"{save_folder}/rbo_{sentence.replace(' ', '_')}_heatmap.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


############################################### SCRIPT #########################################################################
writePairwiseToCSV("rbo")
# writeCentristSimToCSV("rbo")
plotPairwiseSimilarities(ideologies, ideologyWiseList, "rbo")
plotPairwiseSimilaritiesPerSentence(ideologies, ideologyWiseList, "rbo")

# writeBertScores()
# plotBert()
# getPairwiseWilcoxon()
# plotWilcoxon()
# getPairwiseSpearmans()
# plotPairwiseSpearmans()

# getPairwiseKendalls()

# plotPairwiseKendalls()
