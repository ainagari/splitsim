import os
import pandas as pd
from scipy.stats import spearmanr
import ast
import csv
from nltk.corpus import wordnet as wn
from collections import Counter
import argparse
from flota import FlotaTokenizer
from ast import literal_eval
import numpy as np
from utils import load_splitsim
from copy import copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monopoly", default="poly", help="mono or poly")
    parser.add_argument("--pos", default="n", help="n or v")
    args = parser.parse_args()

    original_dataset = load_splitsim(args.monopoly, args.pos)

    predictions_dir = "predictions/" + args.monopoly + "_" + args.pos + "/"
    results_dir = "results/" + args.monopoly + "_" + args.pos + "/"

    ####################################
    # FIRST, DETERMINE SOME CLASSES that are not dependent on the model tested #
    ####################################
    # Include frequency band information here
    print("Getting frequency information")
    in_freq_interval_225375 = []
    interval_size = 0.25
    higher_than_interval = dict()
    lower_than_interval = dict()
    thresholds =[1.5, 1.75, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25]
    for t in thresholds:
        lower_than_interval[t] = []
        higher_than_interval[t] = []

    for i, r in original_dataset.iterrows():
        # frequency interval to create balanced split-sim
        if (2.25 <= r["freq1"] < 3.75) and (2.25 <= r["freq2"] < 3.75):
            in_freq_interval_225375.append(True)
        else:
            in_freq_interval_225375.append(False)

        delta = abs(r['freq1'] - r['freq2'])
        avgfreq = np.average([r['freq1'], r['freq2']])

        for t in thresholds:
            if avgfreq <= t and delta <= 1:
                lower_than_interval[t].append(True)
                higher_than_interval[t].append(False)
            elif avgfreq > t+interval_size and delta <= 1:
                lower_than_interval[t].append(False)
                higher_than_interval[t].append(True)
            else: # out of interval
                lower_than_interval[t].append(False)
                higher_than_interval[t].append(False)

    original_dataset['in_freq_interval_225375'] = in_freq_interval_225375

    for t in thresholds:
        original_dataset['freq_higher_interval_' + str(t)] = higher_than_interval[t]
        original_dataset['freq_lower_interval_' + str(t)] = lower_than_interval[t]

    print("Getting additional info...")

    short_model_names = [col.split("_")[0] for col in original_dataset.columns if "split_class" in col]
    for smn in short_model_names:
        if smn == "fasttext": # number of pieces irrelevant for fasttext
            continue
        number_target_pieces = []
        for i, r in original_dataset.iterrows():
            number_target_pieces.append(len(ast.literal_eval(r[smn + "_split_word1"])) + len(ast.literal_eval(r[smn + "_split_word2"])))
        original_dataset[smn + "_number_target_pieces"] = number_target_pieces

    all_results = []

    # load the flota tokenizer
    tokenizer = FlotaTokenizer("bert-base-uncased", k=3, strict=False, mode="flota")

    # we load the fasttext predictions to be used as "gold standard" (experiments in the Appendix)
    fasttext_predictions_fn = predictions_dir + "fasttext-l0_average_predictions.tsv"
    fasttext_pred = pd.read_csv(fasttext_predictions_fn, sep="\t")
    fasttext_pred = fasttext_pred.drop(columns=["Unnamed: 0"])
    fasttext_pred = fasttext_pred.rename(columns={'predictions': 'fasttext_similarity'})

    ###### Calculate correlations for every single model

    print("Loading predictions")
    for pred_filename in os.listdir(predictions_dir):
        print(pred_filename)
        rep_type, strategy, _ = pred_filename.split("_")
        rep_type = rep_type.split("-")
        layer = int(rep_type[-1][1:])
        rep_type = "-".join(rep_type[:-1])
        predictions = pd.read_csv(predictions_dir+pred_filename, sep="\t")

        predictions_dataset = pd.merge(original_dataset, predictions, on=["word1", "word2"])
        predictions_dataset = pd.merge(fasttext_pred, predictions_dataset, on=["word1", "word2"])

        # Correlations by split type, and global correlations (all)
        # Models with different tokenizations are evaluated on different parts of the dataset.

        if "xlnet" in pred_filename:
            short_model_name = "xlnet"
        elif "electra" in pred_filename:
            short_model_name = "electra"
        else:
            short_model_name = "bert"

        split_class_variable_name = short_model_name + "_split_class"
        word1split_variable_name = short_model_name + "_split_word1"
        word2split_variable_name = short_model_name + "_split_word2"

        for split_type in ['0-split','1-split','2-split', 'all']:
            # First get the correlation using all pairs available in this split-type
            if split_type != "all":
                subset = predictions_dataset[predictions_dataset[split_class_variable_name] == split_type]
            else:
                subset = predictions_dataset
            r, p = spearmanr(subset['wup_similarity'], subset['predictions']) 
            item = {"layer": layer, "representation_type": rep_type, "strategy": strategy, "split-type": split_type, "r": r, "p": p}

            # treating fasttext as the gold standard
            r, p = spearmanr(subset['fasttext_similarity'], subset['predictions']) 
            item['fasttext-all-GS_r'] = r

            # Correlation on subsets in the narrower frequency ranges (BALANCED SPLIT-SIM)
            for freqintervalname in ['in_freq_interval_225375']:
                new_subset = subset[subset[freqintervalname] == True]
                r, p = spearmanr(new_subset['wup_similarity'], new_subset['predictions'])
                item[freqintervalname + "_r"] = r

            for t in thresholds:
                freqnames = ["freq_lower_interval_" + str(t), "freq_higher_interval_" + str(t)]
                for freqname in freqnames:
                    new_subset = subset[subset[freqname] == True]
                    r, p = spearmanr(new_subset['wup_similarity'], new_subset['predictions'])
                    item[freqname + "_r"] = r

            # FOR FLOTA ONLY: separate words that are incomplete after flota tokenization from words that aren't, and calculate correlations separately
            if "flota" in pred_filename and split_type in ['1-split','2-split']:
                subset_bothcomplete_flota = []
                subset_oneincomplete_flota = []
                for i, r in subset.iterrows():
                    tokw1 = tokenizer.tokenize(str(r['word1']))
                    tokw2 = tokenizer.tokenize(str(r['word2']))
                    recw1 = "".join([w.strip("#") for w in tokw1])
                    recw2 = "".join([w.strip("#") for w in tokw2])
                    if recw1 == str(r['word1']) and recw2 == str(r['word2']):
                        subset_bothcomplete_flota.append(r)
                    else:
                        subset_oneincomplete_flota.append(r)

                subset_bothcomplete_flota = pd.DataFrame(subset_bothcomplete_flota)
                subset_oneincomplete_flota = pd.DataFrame(subset_oneincomplete_flota)

                if not subset_bothcomplete_flota.empty:
                    r, p = spearmanr(subset_bothcomplete_flota['wup_similarity'], subset_bothcomplete_flota['predictions'])
                    item["bothcomplete_flota_r"] = r
                else:
                    item["bothcomplete_flota_r"] = np.nan

                if not subset_oneincomplete_flota.empty:
                    r, p = spearmanr(subset_oneincomplete_flota['wup_similarity'], subset_oneincomplete_flota['predictions'])
                    item["oneincomplete_flota_r"] = r
                else:
                    item["oneincomplete_flota_r"] = np.nan


            all_results.append(item)

    all_results = pd.DataFrame(all_results)
    all_results.to_csv(results_dir + "all_results.tsv", sep="\t", index=False)



