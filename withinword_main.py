
import torch
from transformers import BertModel, BertTokenizer, BertConfig, AutoModel, AutoTokenizer, AutoConfig
import pdb
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import numpy as np
import argparse
import sys
from flota import FlotaTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import random
from extract_representations import smart_tokenization
random.seed(9)
try:
    from modeling.character_bert import CharacterBertModel
except ModuleNotFoundError:
    pass
from cb_utils.character_cnn import CharacterIndexer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_wic():
    diri = "WiC"
    pairs = []
    instances = []
    for subset in ["train","dev"]: # put it all together (train and dev)
        fn = diri + subset + "/" + subset + ".data.ids.txt"
        with open(fn) as f:
            for l in f:
                l = l.strip().split("\t")
                p1, p2 = l[3].split("-")
                item = {"id": tuple(l[0].split("-")), "lemma": l[1],'pos':l[2], 'position1':int(p1), "position2":int(p2),
                        "sentence1": l[4], "sentence2": l[5], "word1": l[4].split()[int(p1)], "word2": l[5].split()[int(p2)], "score": l[6], "subset": subset}
                pairs.append(item)

    pairs = pd.DataFrame(pairs)
    instance_ids = set()
    for i, r in pairs.iterrows():
        id1, id2 = r.id
        if id1 not in instance_ids:
            instance1 = dict()
            instance1['lemma'] = r.lemma
            instance1['pos'] = r.pos
            instance1['id'] = id1
            instance1['position'] = r.position1
            instance1['word'] = r.word1
            instance1['sentence'] = r.sentence1
            instance_ids.add(id1)
            instances.append(instance1)
        if id2 not in instance_ids:
            instance2 = dict()
            instance2['lemma'] = r.lemma
            instance2['pos'] = r.pos
            instance2['id'] = id2
            instance2['position'] = r.position2
            instance2['word'] = r.word2
            instance2['sentence'] = r.sentence2
            instance_ids.add(id2)
            instances.append(instance2)

    return pairs, instances


def aggregate_reps(reps_list, strategy):
    if len(reps_list) == 1:
        w, rep = reps_list[0]
        return rep
    else:
        reps = []
        longest_piece = sorted([(w, len(w.strip(token_for_marking_subwords))) for w, rep in reps_list], key=lambda item: item[1])[-1]
        if strategy != "waverage":
            for i, wrep in enumerate(reps_list):
                w, rep = wrep
                if strategy == "average" or (strategy=="longest" and w == longest_piece[0]):
                    reps.append(rep.detach().cpu().numpy())
            rep = np.average(reps, axis=0)

        elif strategy == "waverage":
            clean_subwords = [(subword, len(subword.strip(token_for_marking_subwords))) for subword, rep in reps_list]
            total_len = sum([sw[1] for sw in clean_subwords])
            reps = []
            weights = []
            for (subword, swc), (w, rep) in zip(clean_subwords, reps_list):
                reps.append(rep.detach().cpu().numpy())
                weight = swc / total_len
                weights.append(weight)
            rep = np.average(reps, weights=weights, axis=0)
        else:
            print("NOT IMPLEMENTED")
            sys.exit()

    return rep


def balance_wic_sets(pairs, pair_types):
    balanced_pairs = dict()
    for st in ['1-split','2-split']:
        pairs_this_st = []
        for i, pair in pairs.iterrows():
            this_st = [s for s in ['0-split','1-split','2-split'] if pair['id'] in pair_types[s]]
            assert len(this_st) == 1
            if this_st[0] == st:
                pairs_this_st.append(pair)
        # count by score
        trues = [p for p in pairs_this_st if p['score'] == 'T']
        falses = [p for p in pairs_this_st if p['score'] == 'F']
        random.shuffle(trues)
        balanced_subset = falses + trues[:len(falses)]
        balanced_pairs[st] = [p['id'] for p in balanced_subset]
    return balanced_pairs




def group_ids_by_split(pairs, base_tokenizer, model_name):
    pair_types = {'0-split': [], '1-split': [], '2-split': [], 'all':[]}
    form_types = {'same': [], 'diff': []}
    for i, pair in pairs.iterrows():
        w1, w2 = pair['word1'].lower(), pair['word2'].lower()
        if w1 != w2:
            form_types['diff'].append(pair['id'])
        elif w1 == w2:
            form_types['same'].append(pair['id'])
        if "uncased" in model_name or "character" in model_name or "electra" in model_name:
            tokw1 = base_tokenizer.tokenize(w1)
            tokw2 = base_tokenizer.tokenize(w2)
        elif "cased" in args.model_name:
            tokw1 = base_tokenizer.tokenize(pair['word1'])
            tokw2 = base_tokenizer.tokenize(pair['word2'])
        if len(tokw1) > 1 and len(tokw2) > 1:
            pair_types['2-split'].append(pair['id'])
        elif len(tokw1) == 1 and len(tokw2) == 1:
            pair_types['0-split'].append(pair['id'])
        else:
            pair_types['1-split'].append(pair['id'])
        pair_types['all'].append(pair['id'])
    return pair_types, form_types



def save_predictions(all_predictions, id_to_gold, pair_types, additional_subset_info, predictions_dir):
    preds = []
    for idi in id_to_gold:
        more_info = additional_subset_info[idi]
        for st in ['0-split','1-split','2-split']:
            if idi in pair_types[st]:
                split_type = st
        for layer in range(13):
            item = {'layer':layer,'id':"-".join(idi),"gold":id_to_gold[idi], 'split-type': split_type}
            item['split-type-complex'] = more_info['split-type-complex']
            item['BAL'] = more_info['BAL']
            item['SAME/DIFF'] = more_info['SAME/DIFF']
            for strategy in all_predictions:
                pred = all_predictions[strategy][layer][idi]
                item[strategy] = pred
            preds.append(item)
    preds = pd.DataFrame(preds)

    preds.to_csv(predictions_dir, sep="\t")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="bert-base-uncased", type=str, help="bert-base-uncased, characterbert, google/electra-base-discriminator, xlnet-base-cased")
    parser.add_argument('--path_to_characterbert', type=str,
                        help="path to the folder containing the characterbert model and its config file. Only necessary if args.model_name == 'characterbert'")
    parser.add_argument('--flota', action="store_true", help="whether we use the flota tokenizer. Remember to indicate k too")
    parser.add_argument('--k', default=0, type=int, help="k for Flota. If args.flota is False, this is ignored.")
    parser.add_argument('--lemma', action="store_true", help="whether to replace a word with its lemma in context")
    parser.add_argument('--out_dir', default="withinword_results/")

    args = parser.parse_args()
    
    print(device)

    if args.model_name != "characterbert":
        strategies = ["average", "longest", "waverage"]
    else:
        strategies = ["average"]
    model_name_for_output = ""

    if args.lemma:
        model_name_for_output += "lemma-"
    print("Loading the model")
    if args.model_name == "bert-base-uncased":
        config = BertConfig.from_pretrained(args.model_name, output_hidden_states=True)
        model = BertModel.from_pretrained(args.model_name, config=config)
        base_tokenizer = BertTokenizer.from_pretrained(args.model_name)
        model_name_for_output += "bert"
    elif args.model_name == "characterbert":
        model = CharacterBertModel.from_pretrained(args.path_to_characterbert, output_hidden_states=True)
        config = model.config
        base_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model_name_for_output += "characterbert"
        indexer = CharacterIndexer()
    else:
        config = AutoConfig.from_pretrained(args.model_name, output_hidden_states=True)
        model = AutoModel.from_pretrained(args.model_name, config=config)
        base_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model_name_for_output += args.model_name.replace("/", "#")

    if "xlnet" in args.model_name:
        token_for_marking_subwords = "▁"
    else:
        token_for_marking_subwords = "#"


    if "xlnet" in args.model_name:
        max_length = 100000000000  # xlnet has no sequence length limit
    else:
        max_length = config.max_position_embeddings

    model.to(device)

    if args.flota:
        tokenizer = FlotaTokenizer(args.model_name, k=args.k, strict=False, mode="flota")
        model_name_for_output += "flota-" + str(args.k)
    else:
        tokenizer = base_tokenizer

    print("Loading the task data")
    pairs, instances = load_wic(debugging_mode=args.debugging_mode)

    # Establish 0-, 1-, 2-split classes
    pair_types, form_types = group_ids_by_split(pairs, base_tokenizer, args.model_name)

    balanced_ids = balance_wic_sets(pairs, pair_types)

    additional_subset_info = dict()

    print("Tokenizing, feeding instances to the model and extracting representations")
    for instance in instances:
        # Tokenize
        my_tokens = instance['sentence'].split()
        if args.lemma:
            lemma_to_add = instance['lemma']

            if "xlnet" in args.model_name and my_tokens[instance['position']][0].lower() != my_tokens[instance['position']][0]:
                lemma_to_add = lemma_to_add[0].upper() + lemma_to_add[1:] # uppercasing the lemma for xlnet

            my_tokens[instance['position']] = lemma_to_add

        bert_tokens, map_ori_to_bert, incomplete = smart_tokenization(my_tokens, tokenizer, maxlen=max_length)
        instance["bert_tokens"] = bert_tokens
        instance["map"] = map_ori_to_bert
        instance["bert_target_idcs"] = map_ori_to_bert[instance['position']]

        if incomplete: # it was never the case
            pdb.set_trace()

        # Obtain contextualized representations
        model.eval()
        with torch.no_grad():
            if args.model_name != "characterbert":
                input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(instance['bert_tokens'])])
            elif args.model_name == "characterbert":
                input_ids = indexer.as_padded_tensor([instance['bert_tokens']])

            inputs = {'input_ids': input_ids.to(device)}
            outputs = model(**inputs)

            if "character" in args.model_name:
                hidden_states = outputs[2]
            else:
                hidden_states = outputs.hidden_states

            reps_for_this_instance = dict()
            for occurrence_idx in instance["bert_target_idcs"]:
                w = instance["bert_tokens"][occurrence_idx]
                for l in range(len(hidden_states)):
                    if l not in reps_for_this_instance:
                        reps_for_this_instance[l] = []
                    reps_for_this_instance[l].append((w, hidden_states[l][0][occurrence_idx].cpu()))
            instance["representations"] = reps_for_this_instance

    print("Representations have been extracted")

    print("Calculating similarity values with the different strategies")

    # OBTAIN THE SIMILARITY VALUES FOR DIFFERENT WAYS OF AGGREGATING SPLIT WORDS
    all_predictions = dict()

    for strategy in strategies:
        all_predictions[strategy] = dict() # obtain a cosine sim for each pair
        for layer in range(13):
            all_predictions[strategy][layer] = dict()
            for i, pair in pairs.iterrows():
                id1, id2 = pair['id']
                if pair['id'] in pair_types['0-split'] and strategy != strategies[0]: # no need to recompute predictions of different strategies for 0-split
                    all_predictions[strategy][layer][pair['id']] = all_predictions[strategies[0]][layer][pair['id']]
                else:
                    reps1 = [ins['representations'][layer] for ins in instances if ins['id'] == id1][0]
                    reps2 = [ins['representations'][layer] for ins in instances if ins['id'] == id2][0]
                    rep1 = aggregate_reps(reps1, strategy=strategy)
                    rep2 = aggregate_reps(reps2, strategy=strategy)
                    all_predictions[strategy][layer][pair['id']] = 1 - cosine(rep1, rep2)

    print("Evaluation!!")
    # First get the reference
    id_to_gold = dict()
    all_ids = pairs['id']
    gold = []
    for idi in all_ids:
        pair = pairs[pairs['id'] == idi]
        gold_score = pair['score'].values[0]
        id_to_gold[idi] = gold_score
        gold.append(gold_score)
    results_for_storage = []

    for strategy in all_predictions:
        print("######### " + str(strategy) + " ##########")
        for layer in all_predictions[strategy]:
            print("------ LAYER " + str(layer))
            # first, global...
            predictions = []
            all_ids = pairs['id']
            wic_classification_data = {'0-split-train': [], '0-split-dev': [], '1-split-all': [], '2-split-all': [], 'ALL': [], 'SAME': [],'DIFF': [],
                                       '1-split-SAME':[],
                                       '1-split-DIFF':[], '2-split-SAME':[], '2-split-DIFF': [], '0-split-dev-SAME': [], '0-split-dev-DIFF':[],
                                       '1-split-bal':[],'2-split-bal':[]}
            for idi in all_ids:
                additional_subset_info[idi] = dict()
                prediction = all_predictions[strategy][layer][idi]
                predictions.append(prediction)
                pair = pairs[pairs['id'] == idi]
                gold_score = id_to_gold[idi]

                this_st = [s for s in ['0-split', '1-split', '2-split'] if idi in pair_types[s]]
                this_form = [f for f in form_types.keys() if idi in form_types[f]]
                assert not len(this_st) > 1
                assert not len(this_form) > 1
                this_st = this_st[0]
                this_form = this_form[0]
                # 0-split train
                if 'tr' in idi[1] and this_st == "0-split":
                    wic_classification_data['0-split-train'].append((prediction, gold_score))
                    additional_subset_info[idi]['split-type-complex'] = "0-split-train"
                    additional_subset_info[idi]['BAL'] = 'no bal'
                # 0-split dev
                elif 'de' in idi[1] and this_st == "0-split":
                    wic_classification_data['0-split-dev'].append((prediction, gold_score))
                    wic_classification_data['0-split-dev-' + this_form.upper()].append((prediction, gold_score))
                    additional_subset_info[idi]['split-type-complex'] = "0-split-dev"
                    additional_subset_info[idi]['BAL'] = 'no bal'
                # 1- or 2-split
                else:
                    wic_classification_data[this_st + "-all"].append((prediction, gold_score))
                    wic_classification_data[this_st + "-" + this_form.upper()].append((prediction, gold_score))
                    present_in_bal = idi in balanced_ids[this_st]
                    additional_subset_info[idi]['split-type-complex'] = this_st + "-all"
                    if present_in_bal:
                        wic_classification_data[this_st + "-bal"].append((prediction, gold_score))
                        additional_subset_info[idi]['BAL'] = "bal"
                    else:
                        additional_subset_info[idi]['BAL'] = 'no bal'

                wic_classification_data['ALL'].append((prediction, gold_score)) # to calculate global accuracy
                if idi in form_types['same']:
                    wic_classification_data['SAME'].append((prediction, gold_score))
                elif idi in form_types['diff']:
                    wic_classification_data['DIFF'].append((prediction, gold_score))

                additional_subset_info[idi]['SAME/DIFF'] = this_form.upper()


            Tpredictions = [pred for pred, gold in zip(predictions, gold) if gold == "T"]
            Fpredictions = [pred for pred, gold in zip(predictions, gold) if gold == "F"]

            #print("mean similarity of T instances:", np.round(np.average(Tpredictions),3), "std:", np.round(np.std(Tpredictions), 3))
            #print("mean similarity of F instances:", np.round(np.average(Fpredictions), 3), "std:", np.round(np.std(Fpredictions), 3))

            # TRAINING
            # we train a logreg model on 0-split pairs from the training set and evaluate it on data from the three kinds of split-type
            logreg = LogisticRegression()
            logreg.fit(np.array([x[0] for x in wic_classification_data['0-split-train']]).reshape(-1, 1), [x[1] for x in wic_classification_data['0-split-train']] )

            # PREDICTING AND EVALUATING
            for evaluation_set in wic_classification_data: # evaluating on every subset
                if "train" not in evaluation_set:
                    if not wic_classification_data[evaluation_set]: # this accounts for the 1-split-same case which is empty most of the time (except with xlnet)
                        results_for_storage.append({'strategy': strategy, 'split-type': evaluation_set, "layer": layer,
                             'model': model_name_for_output, "accuracy": np.nan, "goldtrue_predicttrue": [],
                             "goldtrue_predictfalse": [], "goldfalse_predicttrue": [], "goldfalse_predictfalse": []})
                        continue

                    classifier_predictions = logreg.predict(np.array([x[0] for x in wic_classification_data[evaluation_set]]).reshape(-1, 1))

                    gold_eval = [x[1] for x in wic_classification_data[evaluation_set]]
                    accuracy = accuracy_score(classifier_predictions, gold_eval)
                    print("accuracy on", evaluation_set, np.round(accuracy, 2))

                    goldtrue_predicttrue = sum([1 for x, y in zip(classifier_predictions, gold_eval) if x =='T' and y== 'T'])
                    goldfalse_predictfalse = sum([1 for x, y in zip(classifier_predictions, gold_eval) if x == 'F' and y == 'F'])
                    goldtrue_predictfalse = sum([1 for x, y in zip(classifier_predictions, gold_eval) if x == 'F' and y == 'T'])
                    goldfalse_predicttrue = sum([1 for x, y in zip(classifier_predictions, gold_eval) if x == 'T' and y == 'F'])
                    results_for_storage.append({'strategy': strategy, 'split-type': evaluation_set, "layer": layer,
                                                'model': model_name_for_output, "accuracy": accuracy, "goldtrue_predicttrue": goldtrue_predicttrue,
                                                "goldtrue_predictfalse": goldtrue_predictfalse, "goldfalse_predicttrue": goldfalse_predicttrue,"goldfalse_predictfalse": goldfalse_predictfalse})

        for split_type in pair_types:
            if not pair_types[split_type]:
                continue
            ids_this_split = pair_types[split_type]
            predictions = [all_predictions[strategy][layer][idi] for idi in ids_this_split]
            gold = []
            for idi in ids_this_split:
                score = id_to_gold[idi]

                gold.append(score)

    predictions_dir = args.out_dir + "wic_" + model_name_for_output + "_predictions.tsv"
    if not args.debugging_mode:
        results_for_storage = pd.DataFrame(results_for_storage)
        results_for_storage.to_csv(args.out_dir + "wic_" + model_name_for_output + "_results.tsv", sep="\t")
        save_predictions(all_predictions, id_to_gold, pair_types, additional_subset_info, model_name_for_output)







