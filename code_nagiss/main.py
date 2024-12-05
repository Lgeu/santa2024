import argparse
import copy
import gc
import random
from pathlib import Path
from time import time
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, required=True)
    args = parser.parse_args()
    return args


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


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

    def _get_random_neighbor(self, words: list[str]) -> tuple[list[str], tuple]:
        words = words.copy()
        n = len(words)
        op = random.choice(["insert_anywhere", "insert_nearby"])
        if op == "insert_anywhere":
            # 長さ1-5の部分文字列をランダムで選択し、別の場所に挿入
            l = random.randint(1, min(5, n))
            i = random.randint(0, n - l)
            substring = words[i : i + l]
            del words[i : i + l]
            # 新しい挿入位置を選択（どこでも良い）
            j = random.randint(0, n - l)
            words[j:j] = substring
            neighbor_type = ("insert_anywhere", i, j, l)
        elif op == "insert_nearby":
            # 長さ1-2の部分文字列を選択し、-10~+10の位置にランダムに挿入
            l = random.randint(1, min(2, n))
            i = random.randint(0, n - l)
            substring = words[i : i + l]
            del words[i : i + l]
            # 挿入可能な位置の範囲を計算
            min_pos = max(0, i - 10)
            max_pos = min(n - l, i + 10)
            possible_positions = list(range(min_pos, max_pos + 1))
            possible_positions.remove(i)  # 元の位置を除外
            if not possible_positions:
                # 挿入可能な位置がない場合は元に戻す
                words[i:i] = substring
                neighbor_type = ("no_move", i, i, l)
                return words, neighbor_type
            j = random.choice(possible_positions)
            words[j:j] = substring
            neighbor_type = ("insert_nearby", i, j, l)
        return words, neighbor_type

    def _simulated_annealing(
        self,
        words_best: list[str],
        perplexity_best: float,
        idx_target: int,
    ) -> tuple[list[str], float]:
        # 初期状態の設定
        words_current = words_best.copy()
        perplexity_current = perplexity_best
        words_best_global = words_best.copy()
        perplexity_best_global = perplexity_best

        # 焼きなまし法のパラメータ
        T0 = 1000
        T_min = 1e-7
        alpha = 0.99999
        T = T0
        max_iter = 1000000000
        iteration = 0

        pbar = tqdm(total=max_iter)

        while T > T_min and iteration < max_iter:
            # 近傍解の生成
            words_neighbor, neighbor_type = self._get_random_neighbor(words_current)

            # 近傍解の評価
            text_neighbor = " ".join(words_neighbor)
            perplexity_neighbor = self._calc_perplexity(text_neighbor)

            delta_e = perplexity_neighbor - perplexity_current

            if delta_e <= 0:
                # 解を更新
                words_current = words_neighbor.copy()
                perplexity_current = perplexity_neighbor

                # 最良解の更新
                if perplexity_neighbor < perplexity_best_global:
                    words_best_global = words_neighbor.copy()
                    perplexity_best_global = perplexity_neighbor
                    # 結果の保存
                    save_text(
                        self._calc_perplexity, idx_target, text_neighbor, verbose=1
                    )
                    print(
                        f"[Simulated Annealing] Iteration:{iteration} Update: {perplexity_best_global:.2f}"
                    )
            else:
                # 確率的に解を更新
                p_accept = np.exp(-delta_e / T)
                if random.random() < p_accept:
                    words_current = words_neighbor.copy()
                    perplexity_current = perplexity_neighbor

            if iteration % 10000 == 0:
                print(
                    f"[Simulated Annealing] Iteration:{iteration} Best Perplexity: {perplexity_best_global:.2f} Perplexity: {perplexity_current:.2f} Temperature: {T:.2f}"
                )

            # 温度の減少
            T *= alpha
            iteration += 1
            pbar.update(1)

            # 定期的にスコアメモを保存
            if time() > self.last_time_score_memo_saved + 300:
                print("start save_score_memo")
                save_score_memo(self.score_memo, self.score_memo_with_error)
                self.last_time_score_memo_saved = time()
                print("end save_score_memo")

        pbar.close()

    def run(self, idx_target: int):
        words_best, perplexity_best = self._get_best(idx_target)
        self._simulated_annealing(
            words_best,
            perplexity_best,
            idx_target,
        )


if __name__ == "__main__":
    args = parse_args()
    if (args.idx < 0) or (5 < args.idx):
        raise ValueError(f"Invalid idx: {args.idx}")

    path_input_csv = Path("../input/santa-2024/sample_submission.csv")
    path_model = Path("../input/gemma-2/")
    path_save = Path("./save")
    calculator = PerplexityCalculator(model_path=str(path_model))

    calculator.get_perplexity("test")  # ウォームアップ

    optimizer = Optimization(
        calculator,
        path_input_csv,
        path_model,
        path_save,
        flag_use_best=True,
    )

    optimizer.run(args.idx)