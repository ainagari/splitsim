from transformers import BertModel, BertTokenizer, BertConfig, DistilBertModel, DistilBertConfig, AutoTokenizer, AutoModel, AutoConfig
import torch
import pdb
import pickle
import argparse
import sys
from flota import FlotaTokenizer
import pandas as pd
import os
import numpy as np
try:
    from modeling.character_bert import CharacterBertModel
except ModuleNotFoundError:
    pass

from cb_utils.character_cnn import CharacterIndexer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


def smart_tokenization(sentence, tokenizer, maxlen):
    cls_token, sep_token = tokenizer.cls_token, tokenizer.sep_token
    map_ori_to_bert = []
    tok_sent = [cls_token]
    incomplete = False

    for orig_token in sentence:
        if "uncased" in args.model_name or "character" in args.model_name or "electra" in args.model_name:
            orig_token = orig_token.lower()
        else:
            orig_token = orig_token
        current_tokens_bert_idx = [len(tok_sent)]

        # tokenize
        if args.model_name != "characterbert":
            bert_token = tokenizer.tokenize(orig_token)
        elif args.model_name == "characterbert":
            bert_token = tokenizer.basic_tokenizer.tokenize(orig_token)

        ##### check if adding this token to the sequence will result in >= maxlen (=, because there is still [SEP] to add). If so, stop
        if len(tok_sent) + len(bert_token) >= maxlen:
            incomplete = True
            break
        tok_sent.extend(bert_token) # add to my new tokens
        if len(bert_token) > 1: # if the new token has been split
            extra = len(bert_token) - 1
            for i in range(extra):
                current_tokens_bert_idx.append(current_tokens_bert_idx[-1]+1) # list of new positions of the target word in the new tokenization
        map_ori_to_bert.append(current_tokens_bert_idx)

    tok_sent.append(sep_token)

    return tok_sent, map_ori_to_bert, incomplete



def load_selected_sentences(letter, monopoly, pos):
    data = []
    with open("Sentences/selected/selected_sentences_" + monopoly + "_" + pos + ".tsv") as f:
        for l in f:
            l = l.strip().split("\t")
            lemma, pos, position, tokens, pos_tokens = l
            if lemma[0].lower() == letter:
                data.append({"lemma": lemma, "pos":pos, "position": int(position), "tokens": tokens.split(), "pos_tokens": pos_tokens.split()})
    return data


def save_reps_by_lemma(reps, base_fn, no_context, monopoly, pos):
    general_out_dir = "Representations/" + monopoly + "_" + pos + "/"
    if not os.path.exists(general_out_dir):
        os.mkdir(general_out_dir)
    out_dir = general_out_dir + base_fn + "/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if no_context == False: # when using context
        lemmaspos = set([(ins['lemma'], ins['pos']) for ins in reps])
        for lemmapos in lemmaspos:
            rslp = [ins for ins in reps if ins['lemma'] == lemmapos[0] and ins['pos'] == lemmapos[1]]
            lemma, _ = lemmapos
            pickle.dump(rslp, open(out_dir + lemma.lower() + ".pkl", "wb"))
    elif no_context == True:
        for lemmapos in reps:
            lemma, _ = lemmapos
            pickle.dump(reps[lemmapos], open(out_dir + lemma.lower() + ".pkl", "wb"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="bert-base-uncased", type=str, help="alternatives: xlnet-base-cased, google/electra-base-discriminator, characterbert")
    parser.add_argument('--path_to_characterbert', type=str, help="path to the folder containing the characterbert model and its config file. Only necessary if args.model_name == 'characterbert'")
    parser.add_argument('--flota', action="store_true", help="whether we use the flota tokenizer (on all input). Remember to indicate k too")
    parser.add_argument('--k', default=3, type=int, help="k for Flota. If args.flota is False, this is ignored.")
    parser.add_argument('--no_context', action="store_true", help="whether we want to simply extract representations for words out of context.")
    parser.add_argument('--monopoly', type=str, help="mono or poly")
    parser.add_argument('--pos', type=str, help="n (nouns) or v (verbs)")

    args = parser.parse_args()

    if args.model_name == "bert-base-uncased":
        config = BertConfig.from_pretrained(args.model_name, output_hidden_states=True)
        model = BertModel.from_pretrained(args.model_name, config=config)
        base_tokenizer = BertTokenizer.from_pretrained(args.model_name)
        base_fn = "representations_bert"
    elif args.model_name == "characterbert":
        model = CharacterBertModel.from_pretrained(args.path_to_characterbert, output_hidden_states=True)
        config = model.config
        base_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        base_fn = "representations_characterbert"
        indexer = CharacterIndexer()
    else:
        config = AutoConfig.from_pretrained(args.model_name, output_hidden_states=True)
        model = AutoModel.from_pretrained(args.model_name, config=config)
        base_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        base_fn = "representations_" + args.model_name.replace("/", "#")

    if "xlnet" in args.model_name:
        max_length = 100000000000  # xlnet has no sequence length limit
    else:
        max_length = config.max_position_embeddings

    model.to(device)

    if args.flota:
        tokenizer = FlotaTokenizer(args.model_name, k=args.k, strict=False, mode="flota")
        base_fn += "-flota-" + str(args.k)
    else:
        tokenizer = base_tokenizer

    if args.no_context:
        base_fn += "-nocontext"

    # Only get representations of lemmas that are in the dataset
    #########################
    df = pd.read_csv("SPLIT-SIM/dataset_" + args.monopoly + "_" + args.pos + ".tsv", sep="\t")
    all_lemmaspos = set()
    for i, r in df.iterrows():
        all_lemmaspos.add((r['word1'], r['pos1']))
        all_lemmaspos.add((r['word2'], r['pos2']))
    #########################

    for letter in 'qwertyuiopasdfghjklzxcvbnm':
        selected_sentences = load_selected_sentences(letter, args.monopoly, args.pos)

        # If not using sentences
        if args.no_context:
            reps = dict()
            lemmaspos = set([(ins['lemma'], args.pos) for ins in selected_sentences])
            for lemmapos in lemmaspos:
                if lemmapos not in all_lemmaspos:
                    continue
                lemma, pos = lemmapos
                reps[lemmapos] = dict()
                if args.model_name != "characterbert":
                    tokenized = [tokenizer.cls_token] + tokenizer.tokenize(lemma) + [tokenizer.sep_token]
                else:
                    tokenized = [tokenizer.cls_token] + tokenizer.basic_tokenizer.tokenize(lemma) + [tokenizer.sep_token]


                with torch.no_grad():
                    if args.model_name != "characterbert":
                        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized)])
                    elif args.model_name == "characterbert":
                        input_ids = indexer.as_padded_tensor([tokenized])

                    outputs = model(input_ids=input_ids.to(device))

                    if "electra" not in args.model_name:
                        hidden_states = outputs[2]
                    else:
                        hidden_states = outputs.hidden_states

                    for i, occurrence_idx in enumerate(range(1, len(input_ids[0])-1)):
                        w = tokenized[occurrence_idx]
                        reps[lemmapos][(i, w)] = dict()
                        for l in range(len(hidden_states)):
                            reps[lemmapos][(i, w)][l] = torch.tensor(hidden_states[l][0][occurrence_idx].cpu(), dtype=torch.float64)  # 0 is for the batch # i is the subword idx (0 if first subword)

            save_reps_by_lemma(reps, base_fn=base_fn, no_context=True, monopoly=args.monopoly, pos=args.pos)


        # using the sentences
        elif not args.no_context:
            for instance in selected_sentences:
                # tokenize
                bert_tokens, map_ori_to_bert, incomplete = smart_tokenization(instance['tokens'], tokenizer, max_length)

                if incomplete: # if a sentence is longer than the allowed limit, stop and check - the target word might not be included (it didn't happen)
                    print("incomplete instance for word", instance['lemma'], instance['pos'])
                    pdb.set_trace()
                    continue

                instance["bert_tokens"] = bert_tokens
                instance["map"] = map_ori_to_bert
                instance["bert_target_idcs"] = map_ori_to_bert[instance['position']]
                instance["representations"] = dict()

                with torch.no_grad():
                    if args.model_name != "characterbert":
                        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(instance['bert_tokens'])])
                    elif args.model_name == "characterbert":
                        input_ids = indexer.as_padded_tensor([instance['bert_tokens']])

                    inputs = {'input_ids': input_ids.to(device)}
                    outputs = model(**inputs)

                    if "electra" not in args.model_name:
                        hidden_states = outputs[2]
                    else:
                        hidden_states = outputs.hidden_states

                    for i, occurrence_idx in enumerate(instance["bert_target_idcs"]):
                        reps_for_this_instance = dict()
                        w = instance["bert_tokens"][occurrence_idx]
                        for l in range(len(hidden_states)):
                            reps_for_this_instance[l] = torch.tensor(hidden_states[l][0][occurrence_idx].cpu(), dtype=torch.float64)
                        instance["representations"][(i, w)] = reps_for_this_instance

            save_reps_by_lemma(selected_sentences, base_fn=base_fn, no_context=False, monopoly=args.monopoly, pos=args.pos)





