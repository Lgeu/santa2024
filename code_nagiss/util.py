import hashlib
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import heapq

path_input_csv = Path("../input/santa-2024/sample_submission.csv")
path_save = Path("./save")
# path_save.mkdir(parents=True, exist_ok=True)
path_model = Path("../input/gemma-2/")

df_input = pd.read_csv(path_input_csv)


class PriorityQueue:
    def __init__(self, heap):
        self.heap = heap
        heapq.heapify(self.heap)

    def push(self, item):
        heapq.heappush(self.heap, item)

    def pop(self):
        return heapq.heappop(self.heap)

    def pushpop(self, item):
        return heapq.heappushpop(self.heap, item)

    def __call__(self):
        return self.heap

    def __len__(self):
        return len(self.heap)


def get_path_words_best(n_idx):
    path_save_idx = path_save / f"{n_idx:04d}"
    words_original = df_input.loc[n_idx, "text"].split(" ")

    path_txt = path_save_idx.glob("*.txt")
    list_path_txt = list(path_txt)
    if len(list_path_txt) == 0:
        return None, None
    list_scores = [float(path.stem.split("_")[0]) for path in list_path_txt]
    idx_min = np.argmin(list_scores)
    score = list_scores[idx_min]
    path_min = list_path_txt[idx_min]
    text_min = path_min.read_text()
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
    md5 = hashlib.md5(text.encode()).hexdigest()
    path_save_text = path_save_idx / f"{score:.4f}_{md5}.txt"

    with path_save_text.open("w") as f:
        f.write(text)

    return score


def load_score_memo() -> tuple[dict[str, float], dict[str, float]]:
    def load(name: str) -> dict[str, float]:
        path_score_memo = path_save / name
        if path_score_memo.exists():
            with path_score_memo.open("rb") as f:
                score_memo = pickle.load(f)
        else:
            score_memo = {}
        return score_memo

    return load("score_memo.pkl"), load("score_memo_with_error.pkl")


def save_score_memo(
    score_memo: dict[str, float],
    score_memo_with_error: dict[str, float],
):
    def save(name: str, score_memo: dict[str, float]):
        path_score_memo = path_save / name
        with path_score_memo.open("wb") as f:
            pickle.dump(score_memo, f)

    score_memo_original, score_memo_with_error_original = load_score_memo()
    score_memo_original.update(score_memo)
    score_memo_with_error_original.update(score_memo_with_error)
    save("score_memo.pkl", score_memo_original)
    save("score_memo_with_error.pkl", score_memo_with_error_original)


def get_perplexity_(
    scorer,
    score_memo: dict[str, float],
    score_memo_with_error: dict[str, float],
    text: Union[str, list[str]],
) -> Union[float, list[float]]:
    if isinstance(text, str):
        if text in score_memo:
            return score_memo[text]
        score: float = scorer.get_perplexity(text)
        score_memo[text] = score
        return score
    elif isinstance(text, list):
        list_text = text
        list_text_new = []
        for text in list_text:
            if text not in score_memo and text not in score_memo_with_error:
                list_text_new.append(text)

        if len(list_text_new):
            list_score_new: list[float] = scorer.get_perplexity(list_text_new)
            if len(list_text_new) == 1:
                score_memo[list_text_new[0]] = list_score_new[0]
            else:
                for text, score in zip(list_text_new, list_score_new):
                    score_memo_with_error[text] = score
        else:
            list_score_new = []

        list_score = []
        for text in list_text:
            if text in score_memo:
                list_score.append(score_memo[text])
            elif text in score_memo_with_error:
                list_score.append(score_memo_with_error[text])
            else:
                assert False

        return list_score

    else:
        raise ValueError("text should be str or list[str]")
