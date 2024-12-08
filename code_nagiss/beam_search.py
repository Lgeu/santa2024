import copy
import gc
import itertools
import random
from pathlib import Path
from time import time
from typing import Generator, Optional
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from evaluation import PerplexityCalculator
from tqdm.auto import tqdm, trange
from util import (
    get_path_words_best,
    get_perplexity_,
    load_score_memo,
    save_score_memo,
    save_text,
)


class AlphabeticalBeamSearchOptimizer:
    def __init__(
        self,
        path_input_csv: Path,
        path_model: Path,
        path_save: Path,
        path_text: Path,
        beam_width: int = 16,
        num_forseeing: int = 4,
    ):
        self.path_input_csv = Path(path_input_csv)
        self.path_model = Path(path_model)
        self.path_save = Path(path_save)
        self.path_save.mkdir(parents=True, exist_ok=True)
        self.path_text = Path(path_text)
        self.beam_width = beam_width
        self.num_forseeing = num_forseeing

        # データ、スコア計算クラス、スコアメモを読み込む
        self.df = pd.read_csv(self.path_input_csv)
        self.n_idx_total = len(self.df)
        self.calculator = PerplexityCalculator(model_path=str(self.path_model))
        self.score_memo, self.score_memo_with_error = load_score_memo()

        self.text_input = self.path_text.read_text()
        self.text_input = " ".join(self.text_input.split())
        print(f"input text: {self.text_input}")

        self.testcase_idx = None
        for i in range(self.n_idx_total):
            if sorted(self.df.loc[i, "text"].split(" ")) == sorted(
                self.text_input.split(" ")
            ):
                self.testcase_idx = i
                break
        if self.testcase_idx is None:
            print("testcase not found")
            exit()
        print(f"testcase_idx: {self.testcase_idx}")

        self.text_original = self.df.loc[self.testcase_idx, "text"]
        self.words_original = self.text_original.split(" ")
        self.score = self._calc_perplexity(self.text_input)
        print(f"input score: {self.score:.4f}")

    def _calc_perplexity(self, text: str) -> float:
        return get_perplexity_(
            self.calculator, self.score_memo, self.score_memo_with_error, text
        )

    def run(self):
        # get index of alphabetical order
        words = self.text_input.split(" ")

        char_first = words[-1][0]
        idx_first = 0
        for i in range(len(words) - 1, -1, -1):
            if words[i][0] <= char_first:
                idx_first = i
                char_first = words[i][0]
            else:
                break

        print(f"idx_first: {idx_first}")

        list_words = [words[:idx_first]]
        for i in trange(idx_first, len(words)):
            num_foreseeing = min(self.num_forseeing, len(words) - i)
            dict_words2score = defaultdict(lambda: np.inf)

            for words_bef in list_words:
                words_bef = list(words_bef)
                words_unused = self.words_original.copy()
                for word in words_bef:
                    words_unused.remove(word)
                words_unused.sort()
                words_unused = words_unused[:num_foreseeing]

                # all permutations
                list_words_nxt = list(itertools.permutations(words_unused))
                list_words_nxt = [
                    list(words_bef) + list(words_nxt) for words_nxt in list_words_nxt
                ]
                list_score_nxt = [
                    self._calc_perplexity(" ".join(words)) for words in list_words_nxt
                ]
                for words_nxt, score_nxt in zip(list_words_nxt, list_score_nxt):
                    words_nxt_cropped = words_nxt[: i + 1]
                    dict_words2score[tuple(words_nxt_cropped)] = min(
                        dict_words2score[tuple(words_nxt_cropped)], score_nxt
                    )
            list_words = sorted(
                dict_words2score.keys(), key=lambda x: dict_words2score[x]
            )[: self.beam_width]

        text = " ".join(list_words[0])
        score = self._calc_perplexity(text)
        print(f"list_words: {list_words[0]}")
        print(score)

        if score < self.score:
            save_text(self._calc_perplexity, self.testcase_idx, text, verbose=1)

        print("[run] Begin save score_memo")
        save_score_memo(self.score_memo, self.score_memo_with_error)
        print("[run] End save score_memo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_text", type=str, required=True)
    args = parser.parse_args()

    path_input_csv = Path("../input/santa-2024/sample_submission.csv")
    path_model = Path("../input/gemma-2/")
    path_save = Path("./save")
    path_text = Path(args.path_text)
    if not path_text.exists():
        print(f"{path_text} does not exist.")
        exit()
    optimizer = AlphabeticalBeamSearchOptimizer(
        path_input_csv,
        path_model,
        path_save,
        path_text,
        beam_width=16,
        num_forseeing=4,
    )
    optimizer.run()
