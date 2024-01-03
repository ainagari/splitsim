# 💔 Impact of Word Splitting on the Semantic Content of Contextualized Word Representations 🖖🏻

This repository contains data (and soon code) for the paper:

Aina Garí Soler, Matthieu Labeau and Chloé Clavel (2024). Impact of Word Splitting on the Semantic Content of Contextualized Word Representations. To appear in Transactions of the Association for Computational Linguistics (TACL).


### The SPLIT-SIM dataset

The dataset is found in the `SPLIT-SIM/` directory, separated into four `.tsv` files, one for each type of word (monosemous/polysemous nouns/verbs). Files contain word pairs together with their wup similarity, their split-type according to different language models, and the frequency and tokenization of each word in a pair.

### Obtaining contextualized word representations from sentences

Contextualized word representations can be obtained using the script `extract_representations.py`. For example, to obtain BERT representations for monosemous nouns you can use the command:

`python extract_representations.py --model_name bert-base-uncased --monopoly mono --pos n`

Other arguments are:
* `--path_to_characterbert` (path to the folder containing the characterbert model and its config file. Only necessary if `args.model_name == 'characterbert'`)
* `--flota` if the Flota tokenizer is used and `--k` followed by the desired value of this parameter (3 by default)
* `--no_context` to simply extract representations for words out of context

**The sentences used in our experiments** can be found [here](https://drive.google.com/file/d/1yMJisCWTL2JSYt0RYDNp0VSEJQLh0q1h/view?usp=drive_link). The `Sentences/` folder needs to be placed in the same directory as the script.

After running `extract_representations.py`, the embeddings will be saved under the `Representations/` directory, in the corresponding dataset folder, with one pickle file per lemma.

Note that in order to use Flota we have adapted the `flota.py` script from [the original FLOTA repository](https://github.com/valentinhofmann/flota). For CharacterBERT, we use the scripts in the directories `modeling/` and `cb_utils/` (originally called `utils`) downloaded from [the CharacterBERT repository](https://github.com/helboukkouri/character-bert). In order to use CharacterBERT, we needed to set up a separate virtual environment following the instructions in the corresponding repository.



#### Contact

For any questions or requests feel free to contact me: aina dot garisoler at telecom-paris dot fr
