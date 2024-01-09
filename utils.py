from string import punctuation
from nltk.corpus import wordnet as wn
import pandas as pd


def has_punct_or_numbers(word):
    for char in word:
        if char in punctuation or char in '0987654321':
            return True
    return False



# for polysemous words, we calculate the similarities between all sense pairings
# and keep the highest one

def calculate_sim_between_ss(ss1, ss2, pos, similarity_name):
    similarity_functions = {'wup': wn.wup_similarity, 'path': wn.path_similarity, 'lch': wn.lch_similarity}
    similarity_function = similarity_functions[similarity_name]
    sims = []    
    for s1 in ss1:
        for s2 in ss2:
            if s1.pos() == s2.pos() == pos:                
                sims.append(similarity_function(s1, s2))
   
    return max(sims)
   
                

def load_splitsim(monopoly, pos):
    fn = "SPLIT-SIM/dataset_" + monopoly + "_" + pos + ".tsv"
    df = pd.read_csv(fn, sep="\t", keep_default_na=False) 

    if "wup_similarity" in df.columns:
        df['wup_similarity'] = df['wup_similarity'].astype(float)
    if 'predictions' in df.columns:
        df['predictions'] = df['predictions'].astype(float)
    if "freq1" in df.columns and "freq2" in df.columns:
        df['freq1'] = df['freq1'].astype(float)
        df['freq2'] = df['freq2'].astype(float)
    return df
    
def load_splitsim_freqcontrol(monopoly, pos):
    fn = "SPLIT-SIM/freqcontrol_dataset_" + monopoly + "_" + pos + ".tsv"
    df = pd.read_csv(fn, sep="\t") 

    if "wup_similarity" in df.columns:
        df['wup_similarity'] = df['wup_similarity'].astype(float)
    if 'predictions' in df.columns:
        df['predictions'] = df['predictions'].astype(float)
    if "freq1" in df.columns and "freq2" in df.columns:
        df['freq1'] = df['freq1'].astype(float)
        df['freq2'] = df['freq2'].astype(float)
    return df
    
