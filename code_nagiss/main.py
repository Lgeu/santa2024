import copy
import gc
import itertools
import random
from pathlib import Path
from time import time
from typing import Generator, Optional

import numpy as np
import pandas as pd
import torch
from evaluation import PerplexityCalculator
from tqdm.auto import tqdm
from util import (
    PriorityQueue,
    get_path_words_best,
    get_perplexity_,
    load_score_memo,
    save_score_memo,
    save_text,
)

# %%

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def make_neighbors(
    words: list[str],
) -> Generator[tuple[list[str], tuple], None, None]:
    words = words.copy()
    found = {tuple(words)}

    sorted_segments = []
    for i, (left_word, right_word) in enumerate(zip(words, words[1:])):
        if left_word <= right_word:
            if sorted_segments and sorted_segments[-1][1] == i + 1:
                sorted_segments[-1][1] = i + 2
            else:
                sorted_segments.append([i, i + 2])
    sorted_segments = [
        (left, right) for left, right in sorted_segments if right - left >= 4
    ]

    for length in range(1, 5):
        if length >= 2:
            # 区間を既にソートされている部分に入れる
            results = []
            for source_l in range(len(words) - length + 1):
                source_r = source_l + length
                for target_l, target_r in sorted_segments:
                    if source_r <= target_l:
                        permuted = (
                            words[:source_l]
                            + words[source_r:target_l]
                            + sorted(
                                words[source_l:source_r] + words[target_l:target_r]
                            )
                            + words[target_r:]
                        )
                    elif target_r <= source_l:
                        permuted = (
                            words[:target_l]
                            + sorted(
                                words[target_l:target_r] + words[source_l:source_r]
                            )
                            + words[target_r:source_l]
                            + words[source_r:]
                        )
                    else:
                        continue
                    if (t := tuple(permuted)) not in found:
                        found.add(t)
                        results.append(
                            (permuted, (source_l, source_r, target_l, target_r, 3))
                        )
            random.shuffle(results)
            yield from results

        r = range(length, len(words) - length + 1)
        for center in random.sample(r, len(r)):
            results = []
            # 右が短い
            right = center + length
            for left_length in itertools.count(length):
                left = center - left_length
                if left < 0:
                    break
                permuted = (
                    words[:left]
                    + words[center:right]
                    + words[left:center]
                    + words[right:]
                )
                if (t := tuple(permuted)) not in found:
                    found.add(t)
                    results.append((permuted, (left, center, right, 0)))
                if length == 2:
                    permuted = (
                        words[:left]
                        + words[center:right][::-1]
                        + words[left:center]
                        + words[right:]
                    )
                    if (t := tuple(permuted)) not in found:
                        found.add(t)
                        results.append((permuted, (left, center, right, 1)))
                    if left_length == 2:
                        permuted = (
                            words[:left]
                            + words[center:right]
                            + words[left:center][::-1]
                            + words[right:]
                        )
                        if (t := tuple(permuted)) not in found:
                            found.add(t)
                            results.append((permuted, (left, center, right, 2)))
            # 左が短い
            left = center - length
            for right_length in itertools.count(length + 1):
                right = center + right_length
                if right > len(words):
                    break
                permuted = (
                    words[:left]
                    + words[center:right]
                    + words[left:center]
                    + words[right:]
                )
                if (t := tuple(permuted)) not in found:
                    found.add(t)
                    results.append((permuted, (left, center, right, 0)))
                if length == 2:
                    permuted = (
                        words[:left]
                        + words[center:right]
                        + words[left:center][::-1]
                        + words[right:]
                    )
                    if (t := tuple(permuted)) not in found:
                        found.add(t)
                        results.append((permuted, (left, center, right, 1)))
            random.shuffle(results)
            yield from results


class Optimization:
    def __init__(
        self,
        calculator: PerplexityCalculator,
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
        self.calculator = calculator
        self.score_memo, self.score_memo_with_error = load_score_memo()
        self.last_time_score_memo_saved = time()

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
        self.list_num_kick = [1] * self.n_idx_total

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
        iter_total: int = 2000,
    ) -> tuple[list[str], float]:
        
        pbar = tqdm(total=iter_total, mininterval=30)


        def search(words_best: list[str], perplexity_best: float):
            visited = set()
            pqueue = PriorityQueue([(perplexity_best, words_best, 0, [])])
            threshold = 1.015
            max_depth = 0

            for _ in itertools.count(0):
                while True:
                    if len(pqueue) == 0:
                        return None, None, max_depth, None

                    perplexity_nxt_with_error, words_nxt, depth, neighbor_types = pqueue.pop()

                    if tuple(words_nxt) in visited:
                        continue
                    visited.add(tuple(words_nxt))

                    if perplexity_nxt_with_error >= perplexity_best * threshold:
                        return None, None, max_depth, None
                    max_depth = max(max_depth, depth)
                    break

                if perplexity_nxt_with_error < perplexity_best + 2.0:
                    perplexity_nxt = self._calc_perplexity(" ".join(words_nxt))
                else:
                    perplexity_nxt = perplexity_nxt_with_error

                if perplexity_nxt < perplexity_best:
                    return words_nxt, perplexity_nxt, depth, neighbor_types
                
                neighbors = make_neighbors(words_nxt)

                list_words_nxt: list[list[str]] = []
                list_texts_nxt: list[str] = []
                list_neighbor_type: list = []

                while len(list_words_nxt) < 128 * 8:
                    try:
                        words_nxt, neighbor_type = next(neighbors)
                        if tuple(words_nxt) in visited:
                            continue
                        list_words_nxt.append(words_nxt)
                        list_texts_nxt.append(" ".join(words_nxt))
                        list_neighbor_type.append(neighbor_type)
                    except StopIteration:
                        break

                if len(list_words_nxt) == 0:
                    continue

                list_perplexity_nxt_with_error = self._calc_perplexity(list_texts_nxt)
                for words_nxt, perplexity_nxt_with_error, neighbor_type in zip(
                    list_words_nxt,
                    list_perplexity_nxt_with_error,
                    list_neighbor_type,
                ):
                    pqueue.push((perplexity_nxt_with_error, words_nxt, depth + 1, [neighbor_type] + neighbor_types))

                if pbar.n >= iter_total:
                    return None, None, max_depth, None
                if pbar.n % 100 == 0:
                    print(
                        f"[hillclimbing] iter:{pbar.n} best:{perplexity_best:.2f}"
                        f" nxt:{perplexity_nxt:.2f}"
                        f" neighbor:{neighbor_type}"
                        f" depth:{depth}"
                    )
                pbar.update(1)

        words_nxt, perplexity_nxt, depth, neighbor_types = search(words_best, perplexity_best)

        if perplexity_nxt is not None:
            assert perplexity_nxt < perplexity_best
            print(
                f"[hillclimbing] Update: {perplexity_best:.2f}"
                f" -> {perplexity_nxt:.2f},"
                f" neighbor:{','.join(map(str, neighbor_types))}"
                f" depth:{depth}"
            )
            perplexity_best = perplexity_nxt
            words_best = words_nxt
        else:
            print(f"[hillclimbing] No update, depth:{depth}")

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
        n_kick = n_kick - 1
        return n_kick, flag_reset

    # def ILS_kick(
    #     self, words: list[str], n_kick: int = 2
    # ) -> tuple[list[str], list[int]]:
    #     neighbor_types = []
    #     for _ in range(n_kick):
    #         r0 = random.randint(0, len(words) - 1)
    #         r1 = random.randint(0, len(words) - 1)
    #         words[r0], words[r1] = words[r1], words[r0]
    #         neighbor_types.append((r0, r1))
    #     return words, neighbor_types

    def ILS_kick(self, words: list[str], n_kick: int = 2) -> tuple[list[str], list[int]]:
        # double bridge kick
        words_orig = words.copy()
        neighbor_types = []
        for _ in range(n_kick):
            r0, r1, r2, r3 = random.sample(range(1, len(words)), 4)
            r0, r1, r2, r3 = sorted([r0, r1, r2, r3])
            neighbor_types.append((r0, r1, r2, r3))
            words = words[:r0] + words[r2:r3] + words[r1:r2] + words[r0:r1] + words[r3:]
            break # TODO: n_kick = 1 for now
        assert sorted(words) == sorted(words_orig)
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
                iter_total=5000,
            )
            print(f"[run] n_idx:{n_idx} perplexity_best:{perplexity_best:.2f}")
            did_kick = False
            if perplexity_best_old == perplexity_best:
                n_kick, flag_reset = self._calc_n_kick_and_reset(n_idx)
                self.list_num_kick[n_idx] += 1
                did_kick = True
                if flag_reset:
                    print("[run] Reset words")
                    words_best = self._get_best_all(n_idx)[0]
                words_best, neighbor_types = self.ILS_kick(words_best, n_kick=n_kick)
                print(f"[run] Apply {n_kick} kicks: {neighbor_types}")
                perplexity_best = self._calc_perplexity(" ".join(words_best))
            self.list_words_best[n_idx] = words_best
            self.list_perplexity_best[n_idx] = perplexity_best
            self._update_best_all(n_idx, words_best, perplexity_best)
            if not did_kick and perplexity_best < self._get_best_all(n_idx)[1] * 1.1:
                save_text(self._calc_perplexity, n_idx, " ".join(words_best), verbose=1)
            if time() > self.last_time_score_memo_saved + 600:
                save_score_memo(self.score_memo, self.score_memo_with_error)
                self.last_time_score_memo_saved = time()

# %%
if __name__ == "__main__":
    path_input_csv = Path("../input/santa-2024/sample_submission.csv")
    path_model = Path("../input/gemma-2/")
    path_save = Path("./save")
    calculator = PerplexityCalculator(model_path=str(path_model))
    # %%
    calculator.get_perplexity("test") # warm up, needed for kibuna's wsl2
    # %%
    optimizer = Optimization(calculator, path_input_csv, path_model, path_save)

    # %%
    optimizer.run([5])

# %%
