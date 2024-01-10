# 💔 Impact of Word Splitting on the Semantic Content of Contextualized Word Representations 🖖🏻

This repository contains data and code for the paper:

Aina Garí Soler, Matthieu Labeau and Chloé Clavel (2024). Impact of Word Splitting on the Semantic Content of Contextualized Word Representations. To appear in Transactions of the Association for Computational Linguistics (TACL).

## Inter-word experiments

### The SPLIT-SIM dataset

The dataset is found in the `SPLIT-SIM/` directory, separated into four `.tsv` files, one for each type of word (monosemous/polysemous nouns/verbs). Files contain word pairs together with their wup similarity, their split-type according to different language models, and the frequency and tokenization of each word in a pair.
### Obtaining representations, calculating similarities and correlations

To facilitate replication, we share the code to:

1. Extract contextyalized word representations from sentences
2. Make similarity predictions with different pooling strategies
3. Calculate correlations between cosine similarities and wup similarity

But the obtained predictions and correlations are already included in `predictions/[MONOPOLY]_[POS]/` and `results/[MONOPOLY]_[POS]/` and can be analyzed directly with the Jupyter notebook `results_plots_SPLITSIM.ipynb`.


#### Obtaining contextualized word representations from sentences

Contextualized word representations can be obtained using the script `extract_representations.py`. For example, to obtain BERT representations for monosemous nouns you can use the command:

`python extract_representations.py --model_name bert-base-uncased --monopoly mono --pos n`

Other arguments are:
* `--path_to_characterbert` (path to the folder containing the characterbert model and its config file. Only necessary if `args.model_name == 'characterbert'`)
* `--flota` if the Flota tokenizer is used and `--k` followed by the desired value of this parameter (3 by default)
* `--no_context` to simply extract representations for words out of context

**The sentences used in our experiments** can be found [here](https://drive.google.com/file/d/1yMJisCWTL2JSYt0RYDNp0VSEJQLh0q1h/view?usp=drive_link). The `Sentences/` folder needs to be placed in the same directory as the script.

After running `extract_representations.py`, the embeddings will be saved under the `Representations/` directory, in the corresponding dataset folder, with one pickle file per lemma.

Note that in order to use Flota we have adapted the `flota.py` script from [the original FLOTA repository](https://github.com/valentinhofmann/flota). For CharacterBERT, we use the scripts in the directories `modeling/` and `cb_utils/` (originally called `utils`) downloaded from [the CharacterBERT repository](https://github.com/helboukkouri/character-bert). In order to use CharacterBERT, we needed to set up a separate virtual environment following the instructions in the corresponding repository.


#### Making cosine similarity predictions with different pooling strategies


Once representations have been extracted, in order to calculate cosine similarities for split-sim pairs using different pooling strategies, you can use the script `predict.py` with the following arguments:

* `--embedding_type` ("fasttext" for static embeddings, or name of the model as saved in the `Representations` folder: bert, xlnet-base-cased, characterbert, google#electra-base-discriminator... and their '-nocontext' or '-flota-k' variants)
* `--strategy` (pooling strategy to use. Options: average, longest, waverage, omitfirst, omitlast, avg-omitfirst, avg-omitlast
* `--monopoly` and `--pos` (to indicate the subset to use, as above: mono/poly, n/v)

For example:

`python predict.py --embedding_type bert --strategy average --monopoly poly --pos v`


By default, predictions are saved in the "predictions/[MONOPOLY]_[POS]/" directory as one tsv file per model, strategy and layer.

#### Calculating correlations between cosine similarity and wup similarity

The main correlations are then calculated with the script `calculate_correlations.py`. 
This script takes all predictions available for a given subset (`--monopoly` and `--pos`) and calculates Spearman correlations between models' predictions and wup similarity in different subsets of the data (different split-types, balanced SPLIT-SIM, different frequency ranges...).

By default, results are saved in the `results/[MONOPOLY]_[POS]/` folder. 


### Analyzing results

The analyses and results presented in the paper can be found in the Jupyter notebook `results_plots_SPLITSIM.ipynb`.


## Within-word experiments 

The `WiC/` folder contains a slightly modified version of WiC (with ids and gold labels in a single file). The original WiC dataset can be downloaded from [this website](https://pilehvar.github.io/wic/).

The `withinword_main.py` script takes care of all the steps necessary to obtain predictions and correlations. The `withinword_results/' already contains the files that are generated when running this script.

It can be run as follows:

`python withinword_main.py --model_name bert-base-uncased`

Additional options:

* `--lemma` to replace the target word with its lemma
* `--path_to_characterbert`, `--flota` and `--k` as above
* `--out_dir` folder where similarities and correlations will be saved (by default, `withinword_results/`)

The analyses presented in the paper can be found in Jupyter notebook `results_plots_WiC.ipynb`.


## Contact

For any questions or requests feel free to contact me: aina dot garisoler at telecom-paris dot fr


### Citation

(coming soon)
