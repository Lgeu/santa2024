#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
parser.add_argument("--loop", action="store_true")
args = parser.parse_args()


path_input_csv = Path("../../input/santa-2024/sample_submission.csv")
path_save = Path("./save")
path_save_submissions = path_save / "submissions"
path_save_submissions.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(path_input_csv)


# In[2]:


import numpy as np


def update_submission():
    df_submission = df.copy()

    list_scores_submission = []

    for n_idx in range(len(df)):
        text_original = df.loc[n_idx, "text"]
        words_original = text_original.split(" ")

        path_save_idx = path_save / f"{n_idx:04d}"
        # get all of the txt files
        path_txt = path_save_idx.glob("*.txt")
        list_path_txt = list(path_txt)
        if len(list_path_txt) == 0:
            print(f"no txt files in {path_save_idx}")
            continue
        # get scores
        list_scores = [float(path.stem) for path in list_path_txt]
        # print(list_scores)
        # get min score
        idx_min = np.argmin(list_scores)
        score = list_scores[idx_min]
        print(idx_min, score)
        # get min score path
        path_min = list_path_txt[idx_min]
        # print(path_min)
        # get min score text
        text_min = path_min.read_text()
        print(text_min)
        # get min score words
        words_min = text_min.split(" ")
        assert sorted(words_min) == sorted(words_original)

        df_submission.loc[n_idx, "text"] = text_min

        list_scores_submission.append(score)

    # save

    score_ave = np.mean(list_scores_submission)

    print(f"scores: {list_scores_submission}")
    print(f"average score: {score_ave}")

    # save
    # path_save_submission = path_save / "submission.csv"
    path_save_submission = path_save_submissions / f"submission_{score_ave:.6f}.csv"

    df_submission.to_csv(path_save_submission, index=False)

    return path_save_submission, score_ave


def get_best_score_from_submission():
    list_path_csv = list(path_save_submissions.glob("*.csv"))
    list_score = [float(path.stem.split("_")[-1]) for path in list_path_csv]
    if len(list_score) == 0:
        return None, None
    idx_best = np.argmin(list_score)
    path_best = list_path_csv[idx_best]
    print(f"best score: {list_score[idx_best]}")
    print(f"best path: {path_best}")
    return path_best, list_score[idx_best]


# In[3]:


# path_best, score_best = get_best_score_from_submission()


# In[4]:


import subprocess
import time

if args.force:
    path_submission, score = update_submission()
    print(f"force submit: {path_submission}")
    print(f"force score: {score}")
    cmd = f"kaggle competitions submit -c santa-2024 -f {path_submission} -m 'submit'"
    print(cmd)
    subprocess.run(cmd, shell=True)
    print()

elif args.loop:
    while True:
        # get best csv

        path_best, score_best = get_best_score_from_submission()
        if score_best is None:
            score_best = 1e9
        update_submission()
        path_best_new, score_best_new = get_best_score_from_submission()

        if score_best_new < score_best:
            print("submit new submission")
            print(f"score best: {score_best} -> {score_best_new}")

            # kaggle competitions submit -c santa-2024 -f submission.csv -m "Message"

            cmd = f"kaggle competitions submit -c santa-2024 -f {path_best_new} -m 'submit'"
            print(cmd)
            subprocess.run(cmd, shell=True)

        # break

        time.sleep(60 * 30)

else:
    path_submission, score = update_submission()
    print(f"submision path: {path_submission}")