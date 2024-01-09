import pandas as pd
import pickle
import numpy as np
import sys
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import argparse
from pymagnitudelight import *
import pdb
import csv
from utils import load_splitsim
import fasttext.util


def prepare_representations(all_lemmaspos, embedding_type, strategy="average"):
    if "xlnet" in embedding_type:
        token_for_marking_subwords = "▁"
    else:
        token_for_marking_subwords = "#"

    reps_by_layer_and_lemmapos = dict()
    for lemmapos in all_lemmaspos:
        lemmapos = (str(lemmapos[0]), str(lemmapos[1]))
        reps_by_layer_and_lemmapos[lemmapos] = dict()
        if "/" in lemmapos[0]:
            lp = ("_").join(lemmapos[0].split("/"))
        else:
            lp = lemmapos[0]

        repslemma = pickle.load(open("Representations/" + args.monopoly + "_" + args.pos + "/representations_" + embedding_type + "/" + lp + ".pkl", "rb"))

        if "nocontext" not in embedding_type:
            repslemma = [ins for ins in repslemma[:10]]
            tokenized_lemma = [repslemma[0]['bert_tokens'][p] for p in repslemma[0]['map'][repslemma[0]['position']]]
            layers = repslemma[0]['representations'][list(repslemma[0]['representations'].keys())[0]]
        elif "nocontext" in embedding_type:
            repslemma = [repslemma]
            tokenized_lemma = [sub for idx, sub in repslemma[0]]
            layers = repslemma[0][list(repslemma[0].keys())[0]].keys()

        for layer in layers:
            sentence_reps = []
            for inst in repslemma:
                if "nocontext" not in embedding_type:
                    instance = inst['representations']
                else:
                    instance = inst
                if strategy == "average" or (len(instance) == 1 and "avg-omit" in strategy):
                    instance_rep = np.average([instance[idx_subword][layer].detach().cpu().numpy() for idx_subword in instance], axis=0)
                elif "omit" in strategy and len(instance) > 1:
                    # get the index of the token to omit
                    if "omitfirst" in strategy:
                        token_idx_to_omit = 0
                    elif "omitlast" in strategy:
                        token_idx_to_omit = len(tokenized_lemma) - 1
                    instance_rep = np.average([instance[(j, subword)][layer].detach().cpu().numpy() for j, subword in
                             instance if j != token_idx_to_omit], axis=0)

                elif strategy == "longest":
                    # sort subwords by length (removing # or ▁) and pick the longest one)
                    longest_subword = sorted([(j, subword, len(subword.strip(token_for_marking_subwords))) for j, subword in instance], key=lambda i: i[2])[-1]
                    instance_rep = instance[(longest_subword[0], longest_subword[1])][layer].detach().cpu().numpy() # (idx of longest subword, longest subword string)
                elif strategy == "waverage":
                    clean_subwords = [(j, subword, len(subword.strip(token_for_marking_subwords))) for j, subword in instance]
                    total_len = sum([sw[2] for sw in clean_subwords])
                    reps = []
                    weights = []
                    for idx, subword, swlen in clean_subwords:
                        reps.append(instance[(idx, subword)][layer].detach().cpu().numpy())
                        weight = swlen / total_len
                        weights.append(weight)
                    instance_rep = np.average(reps, weights=weights, axis=0)

                else:
                    print("Unrecognized/unimplemented strategy")
                    sys.exit()

                sentence_reps.append(instance_rep)

                # average over the 10 sentences
                lemmapos_rep = np.average(sentence_reps, axis=0)

                # store them by lemma-pos and layer
                reps_by_layer_and_lemmapos[lemmapos][layer] = lemmapos_rep

    return reps_by_layer_and_lemmapos


def obtain_representations(dataset, embedding_type="bert", strategy="average"):
    all_lemmaspos = set()
    for i, r in dataset.iterrows():
        all_lemmaspos.add((str(r['word1']), ''))
        all_lemmaspos.add((str(r['word2']), ''))

    if embedding_type != "fasttext":
        embedding_type = embedding_type.replace("/","#")
        reps = prepare_representations(all_lemmaspos, embedding_type=embedding_type, strategy=strategy)
    elif embedding_type == "fasttext":
        fasttext.util.download_model('en', if_exists='ignore')
        ft = fasttext.load_model('cc.en.300.bin')
        reps = {w: {0: ft.get_word_vector(w[0])} for w in all_lemmaspos}
    else:
        print('Unrecognized/unimplemented embedding_type')
        sys.exit()

    return reps


def make_predictions(dataset, representations, layer):
    for i, r in dataset.iterrows():
        lp1 = (str(r['word1']), '')
        lp2 = (str(r['word2']), '')
        dataset.loc[i, 'predictions'] = 1 - cosine(representations[lp1][layer], representations[lp2][layer])

    return dataset


def save_predictions(dataset, out_fn):
    smaller_df = dataset[["word1", "word2", "predictions"]]
    smaller_df.to_csv(out_fn, sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_type", default="fasttext, bert, xlnet-base-cased, characterbert, google#electra-base-discriminator... and their '-nocontext' or '-flota-k' variants")
    parser.add_argument("--strategy", default="average, waverage, longest, avg-omitfirst, avg-omitlast")
    parser.add_argument("--monopoly", default="mono", help="mono or poly")
    parser.add_argument("--pos", default="v", help="n or v")
    args = parser.parse_args()

    # Load dataset
    dataset = load_splitsim(args.monopoly, args.pos)

    # Obtain representations for the pairs in the dataset (all layers)
    print("Loading embeddings")
    representations = obtain_representations(dataset, embedding_type=args.embedding_type, strategy=args.strategy)

    layers = representations[list(representations.keys())[0]] if args.embedding_type != "fasttext" else [0]

    # Make similarity predictions for every pair
    for layer in layers:
        predictions_dataset = make_predictions(dataset, representations, layer)
        # Save predictions
        predictions_outdir = "predictions/" + args.monopoly + "_" + args.pos + "/"
        if not os.path.exists(predictions_outdir):
            os.mkdir(predictions_outdir)
        predictions_outfn = predictions_outdir + args.embedding_type + "-l" + str(layer) + "_" + args.strategy + "_predictions.tsv"
        save_predictions(predictions_dataset, out_fn=predictions_outfn)
