import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

path_input_csv = Path("../input/santa-2024/sample_submission.csv")
path_save = Path("./save")
# path_save.mkdir(parents=True, exist_ok=True)
path_model = Path("../input/gemma-2/")

df_input = pd.read_csv(path_input_csv)


def get_path_words_best(n_idx):
    path_save_idx = path_save / f"{n_idx:04d}"
    words_original = df_input.loc[n_idx, "text"].split(" ")

    path_txt = path_save_idx.glob("*.txt")
    list_path_txt = list(path_txt)
    if len(list_path_txt) == 0:
        return None, None
    list_scores = [float(path.stem) for path in list_path_txt]
    # print(list_scores)
    # get min score
    idx_min = np.argmin(list_scores)
    score = list_scores[idx_min]
    # get min score path
    path_min = list_path_txt[idx_min]
    # print(path_min)
    # get min score text
    text_min = path_min.read_text()
    # get min score words
    words_min = text_min.split(" ")
    assert sorted(words_min) == sorted(words_original)

    return score, words_min


def save_text(get_perplexity, n_idx, text, verbose=0):
    path_save_idx = path_save / f"{n_idx:04d}"
    if not path_save_idx.exists():
        path_save_idx.mkdir()
    text_original = df_input.loc[n_idx, "text"]
    words_original = text_original.split(" ")
    words = text.split(" ")
    if sorted(words) != sorted(words_original):
        print(f"words are different: {words} != {words_original}")
        return
    text = " ".join(words)
    score = get_perplexity(text)
    if verbose >= 1:
        print(f"score:{score:.4f}")
    if verbose >= 2:
        print(text)
    path_save_text = path_save_idx / f"{score:.4f}.txt"

    with path_save_text.open("w") as f:
        f.write(text)

    return score


def load_score_memo() -> dict:
    path_score_memo = path_save / "score_memo.pkl"
    if path_score_memo.exists():
        with path_score_memo.open("rb") as f:
            score_memo = pickle.load(f)
    else:
        score_memo = {}
    return score_memo


def save_score_memo(score_memo):
    score_memo_original = load_score_memo()
    score_memo_original.update(score_memo)
    path_score_memo = path_save / "score_memo.pkl"
    with path_score_memo.open("wb") as f:
        pickle.dump(score_memo_original, f)


def get_perplexity_(
    scorer, score_memo: dict, text: Union[str, list[str]]
) -> Union[float, list[float]]:
    if isinstance(text, str):
        if text in score_memo:
            return score_memo[text]
        score: float = scorer.get_perplexity(text)
        score_memo[text] = score
        return score
    elif isinstance(text, list):
        list_score_new = []
        list_text = text
        list_text_new = []
        for t in list_text:
            if t not in score_memo:
                list_text_new.append(t)

        if len(list_text_new) > 0:
            list_score_new: list[float] = scorer.get_perplexity(list_text_new)

        if len(list_text_new) == 1:
            score_memo[list_text_new[0]] = list_score_new[0]

        list_score = []
        for t in list_text:
            if t in score_memo:
                list_score.append(score_memo[t])
            else:
                list_score.append(list_score_new.pop(0))

        return list_score

    else:
        raise ValueError("text should be str or list[str]")
