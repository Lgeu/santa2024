from pathlib import Path
import pandas as pd

path_input_csv = Path("../../input/santa-2024/sample_submission.csv")
path_save = Path("./save")
path_save.mkdir(parents=True, exist_ok=True)
path_model = Path("../../input/gemma-2/")

df_sample = pd.read_csv(path_input_csv)

from evaluation import PerplexityCalculator


# In[3]:


from util import get_path_words_best, save_text
from util import load_score_memo, save_score_memo, get_perplexity_

import copy

import numpy as np
import random


import pickle


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, required=True)

args = parser.parse_args()

path = Path(args.path)
if not path.exists():
    raise ValueError(f"{path} is not found.")

df_submission = pd.read_csv(path)

if df_submission.shape != df_sample.shape:
    raise ValueError("Shape of submission is invalid")

score_memo = load_score_memo()
calculator = PerplexityCalculator(model_path=str(path_model))


def get_perplexity(text):
    return get_perplexity_(calculator, score_memo, text)


# check every row
for n_idx in range(len(df_submission)):
    text_sample = df_sample.iloc[n_idx, 1]
    text_submission = df_submission.iloc[n_idx, 1]
    words_sample = text_sample.split()
    words_submission = text_submission.split()
    if sorted(words_sample) != sorted(words_submission):
        raise ValueError(f"Words are not matched at {n_idx}th row.")

    save_text(get_perplexity, n_idx, " ".join(words_submission), verbose=1)
