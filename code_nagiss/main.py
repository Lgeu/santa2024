import copy
import gc
import itertools
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from evaluation import PerplexityCalculator
from tqdm.auto import tqdm
from util import (
    get_path_words_best,
    get_perplexity_,
    load_score_memo,
    save_score_memo,
    save_text,
)


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def make_neighbor_1(words_input: list[str]) -> list[str]:
    """ランダムに単語を選んでランダムな箇所に挿入"""
    words = words_input.copy()
    idx = random.randint(0, len(words) - 1)
    word = words.pop(idx)
    idx_insert = random.randint(0, len(words))
    words.insert(idx_insert, word)
    return words


def make_neighbor_2(words_input: list[str]) -> list[str]:
    """ランダムに単語の列を選んでランダムな箇所に挿入"""
    words = words_input.copy()
    idx1 = random.randint(0, len(words))
    idx2 = random.randint(0, len(words))
    if idx1 == idx2:
        return None
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    words_mid = words[idx1:idx2]
    words_rest = words[:idx1] + words[idx2:]
    idx_insert = random.randint(0, len(words_rest))
    words_new = words_rest[:idx_insert] + words_mid + words_rest[idx_insert:]
    return words_new


def make_neighbor_3(words_input: list[str]) -> list[str]:
    """ランダムに単語の列を選んで先頭か末尾に移動"""
    words = words_input.copy()
    idx1 = random.randint(1, len(words))
    idx2 = random.randint(1, len(words))
    if idx1 == idx2:
        return None
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    words1 = words[:idx1]
    words2 = words[idx1:idx2]
    words3 = words[idx2:]
    coin = random.randint(1, 2)
    if coin == 1:
        words_new = words1 + words3 + words2
    else:
        words_new = words2 + words1 + words3
    return words_new


def make_neighbor_4(words_input: list[str]) -> list[str]:
    """Rotate"""
    words = words_input.copy()
    idx = random.randint(1, len(words) - 1)
    words_new = words[idx:] + words[:idx]
    return words_new


def make_neighbor_5(words_input: list[str]) -> list[str]:
    """隣接した単語を入れ替える"""
    words = words_input.copy()
    idx = random.randint(0, len(words) - 2)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return words


def make_neighbor_6(words_input: list[str]) -> list[str]:
    """区間を反転する"""
    words = words_input.copy()
    idx1 = random.randint(0, len(words))
    idx2 = (idx1 + random.randint(2, len(words) - 2)) % len(words)
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    words[idx1:idx2] = words[idx1:idx2][::-1]
    return words


def make_neighbor_7(words_input: list[str]) -> list[str]:
    """ランダムな 2 単語を入れ替える"""
    words = words_input.copy()
    idx1 = random.randint(0, len(words) - 1)
    idx2 = (idx1 + random.randint(1, len(words) - 1)) % len(words)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return words


def make_neighbor(
    words_input: list[str], neighbor_prob: dict[int, float]
) -> tuple[list[str], int]:
    """ランダムに操作を行う"""
    words_return = None
    while words_return is None:
        coin = int(
            np.random.choice(list(neighbor_prob.keys()), p=list(neighbor_prob.values()))
        )
        if coin == 1:
            words_return = make_neighbor_1(words_input)
        elif coin == 2:
            words_return = make_neighbor_2(words_input)
        elif coin == 3:
            words_return = make_neighbor_3(words_input)
        elif coin == 4:
            words_return = make_neighbor_4(words_input)
        elif coin == 5:
            words_return = make_neighbor_5(words_input)
        elif coin == 6:
            words_return = make_neighbor_6(words_input)
        elif coin == 7:
            words_return = make_neighbor_7(words_input)
        else:
            raise ValueError("Invalid neighbor function coin")
        if words_return == words_input:
            words_return = None
    assert sorted(words_input) == sorted(words_return)
    return words_return, coin


class Optimization:
    def __init__(
        self,
        path_input_csv: Path,
        path_model: Path,
        path_save: Path,
        flag_use_best=True,  # best を使うかどうか
        flag_shuffle=True,  # best を使わない時にシャッフルするかどうか
    ):
        self.path_input_csv = Path(path_input_csv)
        self.path_model = Path(path_model)
        self.path_save = Path(path_save)
        self.path_save.mkdir(parents=True, exist_ok=True)
        self.flag_use_best = flag_use_best
        self.flag_shuffle = flag_shuffle

        # データ、スコア計算クラス、スコアメモを読み込む
        self.df = pd.read_csv(self.path_input_csv)
        self.n_idx_total = len(self.df)
        self.calculator = PerplexityCalculator(model_path=str(self.path_model))
        self.score_memo, self.score_memo_with_error = load_score_memo()

        # 現在までの最良の解
        self.list_words_best: list[list[str]] = []
        self.list_perplexity_best: list[float] = []
        for idx in range(self.n_idx_total):
            if self.flag_use_best:
                _, list_words = get_path_words_best(idx)
                assert list_words is not None
            else:
                text: str = self.df.iloc[idx, 1]
                list_words = text.split()
                if self.flag_shuffle:
                    random.shuffle(list_words)
            text = " ".join(list_words)
            self.list_words_best.append(list_words.copy())
            score_new = self._calc_perplexity(text)
            self.list_perplexity_best.append(score_new)

            print(f"idx:{idx} score:{score_new:.4f}")

        # 行き詰まった時に戻るためのガチの現在までの最良の解
        self.list_words_best_all = copy.deepcopy(self.list_words_best)
        self.list_perplexity_best_all = copy.deepcopy(self.list_perplexity_best)

        # 初期化
        self.list_no_update_cnt = [0] * self.n_idx_total
        self.list_num_kick = [1] * self.n_idx_total
        self.max_no_update_cnt = 10  # キックするまでのイテレーション数

        # 各遷移の選択確率
        self.neighbor_prob = {
            1: 10.0,
            2: 5.0,
            3: 5.0,
            4: 1.0,
            5: 5.0,
            6: 1.0,
            7: 1.0,
        }
        prob_total = sum(self.neighbor_prob.values())
        for key in self.neighbor_prob:
            self.neighbor_prob[key] = self.neighbor_prob[key] / prob_total

    def _calc_perplexity(self, text: str) -> float:
        return get_perplexity_(
            self.calculator, self.score_memo, self.score_memo_with_error, text
        )

    def _get_best(self, n_idx: int) -> tuple[list[str], float]:
        return self.list_words_best[n_idx], self.list_perplexity_best[n_idx]

    def _update_best_all(self, n_idx: int, words: list[str], perplexity: float):
        if perplexity < self.list_perplexity_best_all[n_idx]:
            self.list_words_best_all[n_idx] = words.copy()
            self.list_perplexity_best_all[n_idx] = perplexity

    def _get_best_all(self, n_idx: int) -> tuple[list[str], float]:
        return self.list_words_best_all[n_idx], self.list_perplexity_best_all[n_idx]

    def _hillclimbing(
        self,
        words_best: list[str],
        perplexity_best: float,
        iter_total: int = 1000,
        n_sample: int = 16,
        verbose: bool = False,
    ) -> tuple[list[str], float]:
        pbar = tqdm(total=iter_total, mininterval=5)

        def search(
            words: list[str], last_words: Optional[list[str]] = None, depth: int = 0
        ) -> tuple[float, list[str], list[int]]:
            depth_to_threshold = {
                0: 1.005,
                1: 1.004,
                2: 1.003,
                3: 1.002,
                4: 1.002,
                5: 1.002,
                6: 1.002,
                7: 1.002,
                8: 1.002,
                9: 1.002,
                10: 1.001,
                11: 1.001,
                12: 1.001,
                13: 1.001,
                13: 1.001,
                14: 1.001,
                15: 1.0,
            }

            max_depth = depth
            for _ in itertools.count(0) if depth == 0 else range(50):
                list_words_nxt: list[list[str]] = []
                list_texts_nxt: list[str] = []
                while len(list_words_nxt) < n_sample:
                    words_nxt, neighbor_type = make_neighbor(words, self.neighbor_prob)
                    if words_nxt == last_words:
                        continue
                    list_words_nxt.append(words_nxt)
                    list_texts_nxt.append(" ".join(words_nxt))
                list_perplexity_nxt_with_error = self._calc_perplexity(list_texts_nxt)
                idx_min = int(np.argmin(list_perplexity_nxt_with_error))
                words_nxt = list_words_nxt[idx_min]
                perplexity_nxt_with_error = list_perplexity_nxt_with_error[idx_min]
                if (
                    perplexity_nxt_with_error < perplexity_best + 2.0
                ):  # Cutoff threshold
                    perplexity_nxt = self._calc_perplexity(" ".join(words_nxt))
                else:
                    perplexity_nxt = perplexity_nxt_with_error

                if perplexity_nxt < perplexity_best:
                    return perplexity_nxt, words_nxt, [neighbor_type], max_depth
                elif perplexity_nxt < perplexity_best * depth_to_threshold[depth]:
                    perplexity_nxt, words_nxt, neighbor_types, max_depth_ = search(
                        words_nxt, words, depth + 1
                    )
                    max_depth = max(max_depth, max_depth_)
                    if perplexity_nxt < perplexity_best:
                        return (
                            perplexity_nxt,
                            words_nxt,
                            [neighbor_type] + neighbor_types,
                            max_depth,
                        )

                if pbar.n >= iter_total:
                    break
                if pbar.n % 200 == 0:
                    print(
                        f"[hillclimbing] iter:{pbar.n} best:{perplexity_best:.2f}"
                        f" nxt:{perplexity_nxt:.2f}"
                        f" neighbor:{neighbor_type}"
                        f" depth:{depth}"
                    )
                pbar.update(1)
            return perplexity_nxt, words_nxt, [neighbor_type], max_depth

        perplexity_nxt, words_nxt, neighbor_types, max_depth = search(words_best)
        if perplexity_nxt < perplexity_best:
            print(
                f"[hillclimbing] Update: {perplexity_best:.2f}"
                f" -> {perplexity_nxt:.2f},"
                f" neighbor:{','.join(map(str, neighbor_types))}"
                f" max_depth:{max_depth}"
            )
            perplexity_best = perplexity_nxt
            words_best = words_nxt
        else:
            print(f"[hillclimbing] No update, max_depth:{max_depth}")

        return words_best, perplexity_best

    def _calc_n_kick_and_reset(self, n_idx) -> tuple[int, bool]:
        """??????????"""
        n_kick: int = self.list_num_kick[n_idx]
        i = 1
        while True:
            if n_kick >= i:
                n_kick -= i
            else:
                break
            i += 1
            if i > 16:
                i = 1
        flag_reset = n_kick == 0 and i >= 2
        n_kick = i - n_kick
        n_kick = int(np.sqrt(n_kick)) - 1
        return n_kick, flag_reset

    def ILS_kick(
        self, words: list[str], n_kick: int = 2
    ) -> tuple[list[str], list[int]]:
        neighbor_types = []
        for _ in range(n_kick):
            words, neighbor_type = make_neighbor(words, self.neighbor_prob)
            neighbor_types.append(neighbor_type)
        return words, neighbor_types

    def run(self, list_idx_target: Optional[list[int]] = None):
        n_idx = 0
        if list_idx_target is None:
            list_idx_target = list(range(self.n_idx_total))
        for n_idx in itertools.cycle(list_idx_target):
            free_memory()
            words_best, perplexity_best_old = self._get_best(n_idx)
            print("#" * 80)
            print(f"[run] n_idx:{n_idx} perplexity_best:{perplexity_best_old:.2f}")
            words_best, perplexity_best = self._hillclimbing(
                words_best,
                perplexity_best_old,
                iter_total=1000,
                n_sample=16,
                verbose=True,
            )
            print(f"[run] n_idx:{n_idx} perplexity_best:{perplexity_best:.2f}")
            did_kick = False
            if perplexity_best_old == perplexity_best:
                self.list_no_update_cnt[n_idx] += 1
                if self.list_no_update_cnt[n_idx] >= self.max_no_update_cnt:
                    n_kick, flag_reset = self._calc_n_kick_and_reset(n_idx)
                    self.list_num_kick[n_idx] += 1
                    did_kick = True
                    self.list_no_update_cnt[n_idx] = 0
                    if flag_reset:
                        print("[run] Reset words")
                        words_best = self._get_best_all(n_idx)[0]
                    words_best, neighbor_types = self.ILS_kick(
                        words_best, n_kick=n_kick
                    )
                    print(f"[run] Apply {n_kick} kicks: {neighbor_types}")
                    perplexity_best = self._calc_perplexity(" ".join(words_best))
            else:
                self.list_no_update_cnt[n_idx] = 0
            self.list_words_best[n_idx] = words_best
            self.list_perplexity_best[n_idx] = perplexity_best
            self._update_best_all(n_idx, words_best, perplexity_best)
            if not did_kick:
                save_text(self._calc_perplexity, n_idx, " ".join(words_best), verbose=1)
            save_score_memo(self.score_memo, self.score_memo_with_error)


if __name__ == "__main__":
    path_input_csv = Path("../input/santa-2024/sample_submission.csv")
    path_model = Path("../input/gemma-2/")
    path_save = Path("./save")
    optimizer = Optimization(path_input_csv, path_model, path_save)
    optimizer.run()
