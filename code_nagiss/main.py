import copy
import gc
import itertools
import math
import random
import warnings
from collections import Counter
from time import time
from typing import Generator, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from constants import (
    DF_INPUT,
    LIST_NUM_WORDS,
    LIST_WORD_TO_ID,
    NUM_PROBLEMS,
    PATH_GEMMA,
    PATH_SAVE,
)
from evaluation import PerplexityCalculator
from pretrain import SantaNet
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from util import (
    get_path_words_best,
    get_perplexity_,
    load_score_memo,
    save_score_memo,
    save_text,
)


class ScoreEstimator:
    def __init__(
        self,
        problem_id: int,
        epoch: int,  # 事前学習のモデルをそのまま読み込む場合のみ使われる
        device: torch.device,
    ):
        self.problem_id = problem_id
        self.length = LIST_NUM_WORDS[problem_id]
        online_model_dir = PATH_SAVE / "online"
        online_model_dir.mkdir(parents=True, exist_ok=True)
        self.online_model_path = online_model_dir / f"model_{problem_id}.pt"
        self.pretrained_model_path = (
            PATH_SAVE / f"pretrain/model_{problem_id}_epoch_{epoch}.pt"
        )
        if self.online_model_path.exists():
            print(f"[ScoreEstimator] Load online model: {self.online_model_path}")
            checkpoint = torch.load(self.online_model_path, map_location=device)
        elif self.pretrained_model_path.exists():
            print(
                f"[ScoreEstimator] Load pretrained model: {self.pretrained_model_path}"
            )
            checkpoint = torch.load(self.pretrained_model_path, map_location=device)
        else:
            warnings.warn(f"[ScoreEstimator] Checkpoint not found: {problem_id}")
            checkpoint = None

        self.word_to_id = LIST_WORD_TO_ID[problem_id]

        self.device = device
        self.model = SantaNet(
            vocab_size=len(self.word_to_id), channels=128, num_blocks=12
        ).to(device)
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.00005)

        # 統計用にデータを貯めておく
        self.buffer_texts = []
        self.buffer_scores = []
        self.buffer_predictions = []
        self.update_count = 0

    def estimate_scores(self, texts: list[str]) -> np.ndarray:
        X_list = []
        for text in texts:
            words = text.split()
            assert len(words) == self.length
            X_list.append([self.word_to_id[w] for w in words])
        X = torch.tensor(X_list, dtype=torch.long, device=self.device)
        with torch.no_grad():
            preds: torch.Tensor = self.model(X).squeeze(-1)  # (B,)
        return preds.detach().cpu().numpy()  # log perplexity

    def update_parameters(self, texts: list[str], scores: list[float]):
        X_list = []
        for text in texts:
            words = text.split()
            assert len(words) == self.length
            X_list.append([self.word_to_id[w] for w in words])
        X = torch.tensor(X_list, dtype=torch.long, device=self.device)
        self.model.train()
        pred: torch.Tensor = self.model(X).squeeze(-1)  # (B,)
        target = torch.tensor(scores, dtype=torch.float, device=self.device).log()
        loss = F.l1_loss(pred, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.eval()

        self.buffer_texts.extend(texts)
        self.buffer_scores.extend(scores)
        self.buffer_predictions.extend(pred.detach().cpu().tolist())
        self.update_count += 1
        if self.update_count % 256 == 0:
            corr, _ = spearmanr(self.buffer_scores, self.buffer_predictions)
            print(f"[ScoreEstimator] Spearman: {corr:.4f}")
            self.buffer_texts.clear()
            self.buffer_scores.clear()
            self.buffer_predictions.clear()

    def save_model(self):
        torch.save(
            {"word_to_id": self.word_to_id, "model": self.model.state_dict()},
            self.online_model_path,
        )


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

    max_length = 2 if len(words) >= 50 else 3
    for length in range(1, max_length + 1):
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
                # if length == 2:
                #     permuted = (
                #         words[:left]
                #         + words[center:right][::-1]
                #         + words[left:center]
                #         + words[right:]
                #     )
                #     if (t := tuple(permuted)) not in found:
                #         found.add(t)
                #         results.append((permuted, (left, center, right, 1)))
                #     if left_length == 2:
                #         permuted = (
                #             words[:left]
                #             + words[center:right]
                #             + words[left:center][::-1]
                #             + words[right:]
                #         )
                #         if (t := tuple(permuted)) not in found:
                #             found.add(t)
                #             results.append((permuted, (left, center, right, 2)))
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
                # if length == 2:
                #     permuted = (
                #         words[:left]
                #         + words[center:right]
                #         + words[left:center][::-1]
                #         + words[right:]
                #     )
                #     if (t := tuple(permuted)) not in found:
                #         found.add(t)
                #         results.append((permuted, (left, center, right, 1)))
            random.shuffle(results)
            yield from results


class Optimization:
    def __init__(
        self,
        flag_use_best=True,  # best を使うかどうか
        flag_shuffle=True,  # best を使わない時にシャッフルするかどうか
    ):
        self.flag_use_best = flag_use_best
        self.flag_shuffle = flag_shuffle

        # データ、スコア計算クラス、スコアメモを読み込む
        self.calculator = PerplexityCalculator(model_path=str(PATH_GEMMA))
        self.score_memo, self.score_memo_with_error = load_score_memo()
        self.last_time_score_memo_saved = time()

        # ScoreEstimator
        self.score_estimators = [
            ScoreEstimator(
                problem_id=problem_id,
                epoch=epoch,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            for problem_id, epoch in enumerate([-1, -1, -1, -1, 20, 11])
        ]

        # 現在までの最良の解
        self.list_words_best: list[list[str]] = []
        self.list_perplexity_best: list[float] = []
        for idx in range(NUM_PROBLEMS):
            if self.flag_use_best:
                _, list_words = get_path_words_best(idx)
                assert list_words is not None
            else:
                text: str = DF_INPUT.iloc[idx, 1]
                list_words = text.split()
                if self.flag_shuffle:
                    random.shuffle(list_words)
            text = " ".join(list_words)
            self.list_words_best.append(list_words.copy())
            score_new = self._calc_perplexity(idx, text)
            self.list_perplexity_best.append(score_new)

            print(f"idx:{idx} score:{score_new:.4f}")

        # 行き詰まった時に戻るためのガチの現在までの最良の解
        self.list_words_best_all = copy.deepcopy(self.list_words_best)
        self.list_perplexity_best_all = copy.deepcopy(self.list_perplexity_best)

        # 初期化
        self.list_num_kick = [1] * NUM_PROBLEMS

        # ビームサーチ
        self.list_population: list[list[str]] = [[] for _ in range(NUM_PROBLEMS)]
        self.list_perplexity_population: list[float] = [np.inf] * NUM_PROBLEMS
        self.word_to_id = LIST_WORD_TO_ID

    def _calc_dist(
        self,
        n_idx: int,
        list_words1: Union[list[str], list[list[str]]],
        list_words2: Union[list[str], list[list[str]]],
    ) -> int:
        if (list_words1 == []) or (list_words2 == []):
            return len(self.list_words_best[n_idx])
        if isinstance(list_words1[0], str):
            list_words1 = [list_words1]
        if isinstance(list_words2[0], str):
            list_words2 = [list_words2]
        list_id1 = [
            [self.word_to_id[n_idx][word] for word in words] for words in list_words1
        ]
        list_id2 = [
            [self.word_to_id[n_idx][word] for word in words] for words in list_words2
        ]
        list_id1 = np.array(list_id1)
        list_id2 = np.array(list_id2)

        # use lcs to compute distance
        # def get_lcs_length(a, b):
        #     a = np.array(a)
        #     b = np.array(b)
        #     equal = a[:, None] == b[None, :]
        #     dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
        #     for i in range(len(a)):
        #         dp[i + 1][1:] = np.maximum(dp[i][1:], dp[i][:-1] + equal[i])
        #         dp[i + 1] = np.maximum.accumulate(dp[i + 1])
        #     return dp[-1][-1]

        # lcs_length_max = 0
        # for id1 in list_id1:
        #     for id2 in list_id2:
        #         lcs_length_max = max(lcs_length_max, get_lcs_length(id1, id2))

        # written by o1
        N1, L1 = list_id1.shape
        N2, L2 = list_id2.shape

        # equal: (N1, N2, L1, L2)
        equal = (list_id1[:, None, :, None] == list_id2[None, :, None, :]).astype(
            np.int16
        )

        dp = np.zeros((N1, N2, L1 + 1, L2 + 1), dtype=np.int16)

        for i in range(L1):
            # dp[:, :, i, 1:] と dp[:, :, i, :-1] は形状 (N1, N2, L2)
            dp[:, :, i + 1, 1:] = np.maximum(
                dp[:, :, i, 1:],  # パス1 (i進めただけ)
                dp[:, :, i, :-1] + equal[:, :, i],  # パス2 (マッチした場合の伸び)
            )
            # 累積最大化 (axis=-1: L2方向)
            dp[:, :, i + 1] = np.maximum.accumulate(dp[:, :, i + 1], axis=-1)

        # 各組み合わせのLCS長
        all_lcs = dp[:, :, -1, -1]

        # lcs_length_max
        lcs_length_max = all_lcs.max()

        dist = len(list_words1[0]) - lcs_length_max
        return dist

    def _calc_perplexity(
        self, n_idx: int, text: Union[str, list[str]]
    ) -> Union[float, list[float]]:
        return get_perplexity_(
            self.calculator, n_idx, self.score_memo, self.score_memo_with_error, text
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
        n_idx: int,
        words_best: list[str],
        perplexity_best: float,
        score_estimator: ScoreEstimator,
        iter_total: int = 500,
    ) -> tuple[list[str], float]:
        pbar = tqdm(total=iter_total, mininterval=30)

        class Stats:
            def __init__(self, max_value: int):
                self.max_value = max_value
                self.accepted = Counter()
                self.rejected = Counter()

            def summary(self) -> str:
                n_bins = 8
                accepted = [0] * n_bins
                for value, count in self.accepted.items():
                    assert 0 <= value < self.max_value
                    accepted[value * n_bins // self.max_value] += count
                rejected = [0] * n_bins
                for value, count in self.rejected.items():
                    assert 0 <= value < self.max_value
                    rejected[value * n_bins // self.max_value] += count
                return (
                    f"accepted:{accepted} rejected:{rejected}"
                    f" total:{sum(accepted) + sum(rejected)}"
                )

        batch_size = 128
        stats = Stats(max_value=batch_size)

        visited = set()

        def search(
            words: list[str], depth: int = 0
        ) -> tuple[float, list[str], list[int]]:
            visited.add(tuple(words))
            if n_idx == 0:
                # 未検証
                depth_to_threshold = {
                    0: 1.2,
                    1: 1.12,
                    2: 1.08,
                    3: 1.06,
                    4: 1.04,
                    5: 1.03,
                    6: 1.025,
                    7: 1.02,
                    8: 1.015,
                    9: 1.01,
                    10: 1.01,
                    11: 1.01,
                    12: 1.005,
                    13: 1.005,
                    14: 1.002,
                    15: 1.002,
                    16: 1.002,
                    17: 1.001,
                    18: 1.001,
                    19: 1.001,
                    20: 1.0,
                }
            elif n_idx in [1, 2]:
                depth_to_threshold = {
                    0: 1.2,
                    1: 1.12,
                    2: 1.08,
                    3: 1.06,
                    4: 1.04,
                    5: 1.03,
                    6: 1.025,
                    7: 1.02,
                    8: 1.015,
                    9: 1.01,
                    10: 1.01,
                    11: 1.01,
                    12: 1.005,
                    13: 1.005,
                    14: 1.002,
                    15: 1.002,
                    16: 1.002,
                    17: 1.001,
                    18: 1.001,
                    19: 1.001,
                    20: 1.0,
                }
            elif n_idx == 3:
                # 未検証
                depth_to_threshold = {
                    0: 1.1,
                    1: 1.06,
                    2: 1.04,
                    3: 1.03,
                    4: 1.02,
                    5: 1.015,
                    6: 1.01,
                    7: 1.008,
                    8: 1.006,
                    9: 1.005,
                    10: 1.004,
                    11: 1.003,
                    12: 1.003,
                    13: 1.002,
                    14: 1.002,
                    15: 1.001,
                    16: 1.001,
                    17: 1.001,
                    18: 1.001,
                    19: 1.001,
                    20: 1.0,
                }
            elif n_idx == 4:
                depth_to_threshold = {
                    0: 1.05,
                    1: 1.03,
                    2: 1.02,
                    3: 1.015,
                    4: 1.01,
                    5: 1.008,
                    6: 1.006,
                    7: 1.004,
                    8: 1.003,
                    9: 1.002,
                    10: 1.002,
                    11: 1.002,
                    12: 1.002,
                    13: 1.002,
                    14: 1.001,
                    15: 1.001,
                    16: 1.001,
                    17: 1.001,
                    18: 1.001,
                    19: 1.001,
                    20: 1.0,
                }
            elif n_idx == 5:
                depth_to_threshold = {
                    0: 1.015,
                    1: 1.01,
                    2: 1.007,
                    3: 1.005,
                    4: 1.004,
                    5: 1.0035,
                    6: 1.003,
                    7: 1.0025,
                    8: 1.002,
                    9: 1.0015,
                    10: 1.001,
                    11: 1.001,
                    12: 1.001,
                    13: 1.001,
                    14: 1.001,
                    15: 1.001,
                    16: 1.001,
                    17: 1.001,
                    18: 1.001,
                    19: 1.001,
                    20: 1.0,
                }
            else:
                raise ValueError(f"Invalid n_idx: {n_idx}")

            neighbors = make_neighbors(words)
            max_depth = depth
            for _ in itertools.count(0):
                list_words_nxt: list[list[str]] = []
                list_texts_nxt: list[str] = []
                list_neighbor_type: list = []

                num_candidates = 2048 if depth < 2 else 4096 if depth < 5 else 8192
                while len(list_words_nxt) < num_candidates:
                    try:
                        words_nxt, neighbor_type = next(neighbors)
                        if tuple(words_nxt) in visited:
                            continue
                        list_words_nxt.append(words_nxt)
                        list_texts_nxt.append(" ".join(words_nxt))
                        list_neighbor_type.append(neighbor_type)
                    except StopIteration:
                        break
                if len(list_words_nxt) < min(
                    num_candidates, int(1.5 * len(words) ** 2)
                ):
                    return None, None, None, max_depth

                # 枝刈り
                # 推定スコア上位 112 個とランダム 16 個を選ぶ ε-greedy
                estimated_scores = score_estimator.estimate_scores(list_texts_nxt)
                indices_sorted = np.argsort(estimated_scores).tolist()
                indices_keep = indices_sorted[:112] + random.sample(
                    indices_sorted[112:], 16
                )
                assert len(indices_keep) == batch_size
                list_words_nxt = [list_words_nxt[i] for i in indices_keep]
                list_texts_nxt = [list_texts_nxt[i] for i in indices_keep]
                list_neighbor_type = [list_neighbor_type[i] for i in indices_keep]

                list_perplexity_nxt_with_error = self._calc_perplexity(
                    n_idx, list_texts_nxt
                )

                score_estimator.update_parameters(
                    list_texts_nxt, list_perplexity_nxt_with_error
                )

                estimated_rank = int(np.argmin(list_perplexity_nxt_with_error))
                words_nxt = list_words_nxt[estimated_rank]
                perplexity_nxt_with_error = list_perplexity_nxt_with_error[
                    estimated_rank
                ]
                neighbor_type = list_neighbor_type[estimated_rank]
                if perplexity_nxt_with_error < perplexity_best + 2.0:
                    perplexity_nxt = self._calc_perplexity(n_idx, " ".join(words_nxt))
                else:
                    perplexity_nxt = perplexity_nxt_with_error

                if perplexity_nxt < perplexity_best:
                    stats.accepted[estimated_rank] += 1
                    return perplexity_nxt, words_nxt, [neighbor_type], max_depth
                elif perplexity_nxt < perplexity_best * depth_to_threshold[depth]:
                    search_order = list(range(batch_size))
                    random.shuffle(search_order)
                    for estimated_rank in search_order:
                        words_nxt = list_words_nxt[estimated_rank]
                        perplexity_nxt = list_perplexity_nxt_with_error[estimated_rank]
                        neighbor_type = list_neighbor_type[estimated_rank]
                        if (
                            perplexity_nxt
                            >= perplexity_best * depth_to_threshold[depth]
                        ):
                            stats.rejected[estimated_rank] += 1
                            continue
                        if tuple(words_nxt) in visited:
                            continue
                        stats.accepted[estimated_rank] += 1
                        perplexity_nxt, words_nxt, neighbor_types, max_depth_ = search(
                            words_nxt, depth + 1
                        )
                        max_depth = max(max_depth, max_depth_)
                        if perplexity_nxt is not None:
                            assert perplexity_nxt < perplexity_best
                            return (
                                perplexity_nxt,
                                words_nxt,
                                [neighbor_type] + neighbor_types,
                                max_depth,
                            )

                if pbar.n >= iter_total:
                    return None, None, None, max_depth
                if pbar.n % 100 == 0:
                    print(
                        f"[hillclimbing] iter:{pbar.n} best:{perplexity_best:.2f}"
                        f" nxt:{perplexity_nxt or math.inf:.2f}"
                        f" neighbor:{neighbor_type}"
                        f" depth:{depth}"
                        f" {stats.summary()}"
                    )
                pbar.update(1)

        perplexity_nxt, words_nxt, neighbor_types, max_depth = search(words_best)
        if perplexity_nxt is not None:
            assert perplexity_nxt < perplexity_best
            print(
                f"[hillclimbing] Update: {perplexity_best:.2f}"
                f" -> {perplexity_nxt:.2f},"
                f" neighbor:{','.join(map(str, neighbor_types))}"
                f" max_depth:{max_depth}"
                f" {stats.summary()}"
            )
            perplexity_best = perplexity_nxt
            words_best = words_nxt
        else:
            print(f"[hillclimbing] No update, max_depth:{max_depth} {stats.summary()}")

        return words_best, perplexity_best

    def _beam_search(
        self,
        n_idx: int,
        score_estimator: ScoreEstimator,
        population_size: int = 16,
        iter_total: int = 1000,
        initial_dist_minimum: Optional[int] = None,
    ):
        if initial_dist_minimum is None:
            initial_dist_minimum = len(self.list_words_best[n_idx]) // 3
        # initialize population
        self.list_population[n_idx] = []
        self.list_perplexity_population[n_idx] = []
        for i in range(population_size):
            while True:
                words = self.list_words_best[n_idx].copy()
                random.shuffle(words)
                dist = self._calc_dist(n_idx, words, self.list_population[n_idx])
                print(f"dist: {dist}\twords: {words}")
                if dist >= initial_dist_minimum:
                    break
            self.list_population[n_idx].append(words)
        self.list_perplexity_population[n_idx] = [
            self._calc_perplexity(n_idx, " ".join(words))
            for words in self.list_population[n_idx]
        ]

        class Stats:
            def __init__(self, max_value: int):
                self.max_value = max_value
                self.accepted = Counter()
                self.rejected = Counter()

            def summary(self) -> str:
                n_bins = 8
                accepted = [0] * n_bins
                for value, count in self.accepted.items():
                    assert 0 <= value < self.max_value
                    accepted[value * n_bins // self.max_value] += count
                rejected = [0] * n_bins
                for value, count in self.rejected.items():
                    assert 0 <= value < self.max_value
                    rejected[value * n_bins // self.max_value] += count
                return (
                    f"accepted:{accepted} rejected:{rejected}"
                    f" total:{sum(accepted) + sum(rejected)}"
                )

        pbar = tqdm(total=iter_total, mininterval=30)

        batch_size = 128
        stats = Stats(max_value=batch_size)

        visited = set()

        perplexity_best = np.inf

        def search(
            words: list[str],
            depth: int = 0,
            distance_minimum: int = 0,
            population_other: list[list[str]] = [],
        ) -> tuple[float, list[str], list[int]]:
            visited.add(tuple(words))
            if n_idx == 0:
                # 未検証
                depth_to_threshold = {
                    0: 1.2,
                    1: 1.12,
                    2: 1.08,
                    3: 1.06,
                    4: 1.04,
                    5: 1.03,
                    6: 1.025,
                    7: 1.02,
                    8: 1.015,
                    9: 1.01,
                    10: 1.01,
                    11: 1.01,
                    12: 1.005,
                    13: 1.005,
                    14: 1.002,
                    15: 1.002,
                    16: 1.002,
                    17: 1.001,
                    18: 1.001,
                    19: 1.001,
                    20: 1.0,
                }
            elif n_idx in [1, 2]:
                depth_to_threshold = {
                    0: 1.2,
                    1: 1.12,
                    2: 1.08,
                    3: 1.06,
                    4: 1.04,
                    5: 1.03,
                    6: 1.025,
                    7: 1.02,
                    8: 1.015,
                    9: 1.01,
                    10: 1.01,
                    11: 1.01,
                    12: 1.005,
                    13: 1.005,
                    14: 1.002,
                    15: 1.002,
                    16: 1.002,
                    17: 1.001,
                    18: 1.001,
                    19: 1.001,
                    20: 1.0,
                }
            elif n_idx == 3:
                # 未検証
                depth_to_threshold = {
                    0: 1.1,
                    1: 1.06,
                    2: 1.04,
                    3: 1.03,
                    4: 1.02,
                    5: 1.015,
                    6: 1.01,
                    7: 1.008,
                    8: 1.006,
                    9: 1.005,
                    10: 1.004,
                    11: 1.003,
                    12: 1.003,
                    13: 1.002,
                    14: 1.002,
                    15: 1.001,
                    16: 1.001,
                    17: 1.001,
                    18: 1.001,
                    19: 1.001,
                    20: 1.0,
                }
            elif n_idx == 4:
                depth_to_threshold = {
                    0: 1.05,
                    1: 1.03,
                    2: 1.02,
                    3: 1.015,
                    4: 1.01,
                    5: 1.008,
                    6: 1.006,
                    7: 1.004,
                    8: 1.003,
                    9: 1.002,
                    10: 1.002,
                    11: 1.002,
                    12: 1.002,
                    13: 1.002,
                    14: 1.001,
                    15: 1.001,
                    16: 1.001,
                    17: 1.001,
                    18: 1.001,
                    19: 1.001,
                    20: 1.0,
                }
            elif n_idx == 5:
                depth_to_threshold = {
                    0: 1.015,
                    1: 1.01,
                    2: 1.007,
                    3: 1.005,
                    4: 1.004,
                    5: 1.0035,
                    6: 1.003,
                    7: 1.0025,
                    8: 1.002,
                    9: 1.0015,
                    10: 1.001,
                    11: 1.001,
                    12: 1.001,
                    13: 1.001,
                    14: 1.001,
                    15: 1.001,
                    16: 1.001,
                    17: 1.001,
                    18: 1.001,
                    19: 1.001,
                    20: 1.0,
                }
            else:
                raise ValueError(f"Invalid n_idx: {n_idx}")

            neighbors = make_neighbors(words)
            max_depth = depth
            for _ in itertools.count(0):
                list_words_nxt: list[list[str]] = []
                list_texts_nxt: list[str] = []
                list_neighbor_type: list = []

                num_candidates = 2048 if depth < 2 else 4096 if depth < 5 else 8192
                while len(list_words_nxt) < num_candidates:
                    try:
                        words_nxt, neighbor_type = next(neighbors)
                        if tuple(words_nxt) in visited:
                            continue
                        if (
                            distance_minimum > 0
                            and self._calc_dist(n_idx, words_nxt, population_other)
                            < distance_minimum
                        ):
                            continue
                        list_words_nxt.append(words_nxt)
                        list_texts_nxt.append(" ".join(words_nxt))
                        list_neighbor_type.append(neighbor_type)
                    except StopIteration:
                        break
                if len(list_words_nxt) < min(
                    num_candidates, int(1.5 * len(words) ** 2)
                ):
                    return None, None, None, max_depth

                # 枝刈り
                # 推定スコア上位 112 個とランダム 16 個を選ぶ ε-greedy
                estimated_scores = score_estimator.estimate_scores(list_texts_nxt)
                indices_sorted = np.argsort(estimated_scores).tolist()
                indices_keep = indices_sorted[:112] + random.sample(
                    indices_sorted[112:], 16
                )
                assert len(indices_keep) == batch_size
                list_words_nxt = [list_words_nxt[i] for i in indices_keep]
                list_texts_nxt = [list_texts_nxt[i] for i in indices_keep]
                list_neighbor_type = [list_neighbor_type[i] for i in indices_keep]

                list_perplexity_nxt_with_error = self._calc_perplexity(
                    n_idx, list_texts_nxt
                )

                score_estimator.update_parameters(
                    list_texts_nxt, list_perplexity_nxt_with_error
                )

                estimated_rank = int(np.argmin(list_perplexity_nxt_with_error))
                words_nxt = list_words_nxt[estimated_rank]
                perplexity_nxt_with_error = list_perplexity_nxt_with_error[
                    estimated_rank
                ]
                neighbor_type = list_neighbor_type[estimated_rank]
                if perplexity_nxt_with_error < perplexity_best + 2.0:
                    perplexity_nxt = self._calc_perplexity(n_idx, " ".join(words_nxt))
                else:
                    perplexity_nxt = perplexity_nxt_with_error

                if perplexity_nxt < perplexity_best:
                    stats.accepted[estimated_rank] += 1
                    return perplexity_nxt, words_nxt, [neighbor_type], max_depth
                elif perplexity_nxt < perplexity_best * depth_to_threshold[depth]:
                    search_order = list(range(batch_size))
                    random.shuffle(search_order)
                    for estimated_rank in search_order:
                        words_nxt = list_words_nxt[estimated_rank]
                        perplexity_nxt = list_perplexity_nxt_with_error[estimated_rank]
                        neighbor_type = list_neighbor_type[estimated_rank]
                        if (
                            perplexity_nxt
                            >= perplexity_best * depth_to_threshold[depth]
                        ):
                            stats.rejected[estimated_rank] += 1
                            continue
                        if tuple(words_nxt) in visited:
                            continue
                        stats.accepted[estimated_rank] += 1
                        perplexity_nxt, words_nxt, neighbor_types, max_depth_ = search(
                            words_nxt, depth + 1, distance_minimum, population_other
                        )
                        max_depth = max(max_depth, max_depth_)
                        if perplexity_nxt is not None:
                            assert perplexity_nxt < perplexity_best
                            return (
                                perplexity_nxt,
                                words_nxt,
                                [neighbor_type] + neighbor_types,
                                max_depth,
                            )

                if pbar.n >= iter_total:
                    return None, None, None, max_depth
                if pbar.n % 100 == 0:
                    print(
                        f"[hillclimbing] iter:{pbar.n} best:{perplexity_best:.2f}"
                        f" nxt:{perplexity_nxt or math.inf:.2f}"
                        f" neighbor:{neighbor_type}"
                        f" depth:{depth}"
                        f" {stats.summary()}"
                    )
                pbar.update(1)

        for dist_minimum in range(initial_dist_minimum, 0, -1):
            print(f"[beam_search] dist_minimum:{dist_minimum}")
            visited = set()
            # search order
            search_order = list(range(population_size))
            cnt_no_updates = [0] * population_size
            max_no_updates = 5
            while min(cnt_no_updates) < max_no_updates:
                print(
                    f"[beam_search] perplexities:{self.list_perplexity_population[n_idx]}"
                )
                random.shuffle(search_order)
                for i in search_order:
                    if cnt_no_updates[i] >= max_no_updates:
                        print(f"[beam_search] Skip i:{i}")
                        continue
                    print(
                        f"[beam_search] Run i:{i}, cnt_no_updates:{cnt_no_updates[i]}, dist_minimum:{dist_minimum}"
                    )
                    perplexity_best = self.list_perplexity_population[n_idx][i]
                    population_other = (
                        self.list_population[n_idx][:i]
                        + self.list_population[n_idx][i + 1 :]
                    )
                    perplexity_nxt, words_nxt, neighbor_types, max_depth = search(
                        self.list_population[n_idx][i],
                        distance_minimum=dist_minimum,
                        population_other=population_other,
                    )
                    if perplexity_nxt is not None:
                        assert perplexity_nxt < perplexity_best
                        print(
                            f"[beam_search] Update: {perplexity_best:.2f}"
                            f" -> {perplexity_nxt:.2f},"
                            f" neighbor:{','.join(map(str, neighbor_types))}"
                            f" max_depth:{max_depth}"
                            f" {stats.summary()}"
                        )
                        perplexity_best = perplexity_nxt
                        self.list_population[n_idx][i] = words_nxt
                        self.list_perplexity_population[n_idx][i] = perplexity_nxt
                        cnt_no_updates[i] = 0

                        # check save
                        if perplexity_nxt < self._get_best_all(n_idx)[1] * 1.1:
                            save_text(
                                self._calc_perplexity,
                                n_idx,
                                " ".join(words_nxt),
                                verbose=1,
                            )
                            self._update_best_all(n_idx, words_nxt, perplexity_nxt)

                    else:
                        print(
                            f"[beam_search] No update, max_depth:{max_depth} {stats.summary()}"
                        )
                        cnt_no_updates[i] += 1

        perplexity_best = min(self.list_perplexity_population[n_idx])
        words_best = self.list_population[n_idx][
            self.list_perplexity_population[n_idx].index(perplexity_best)
        ]
        perplexity_best = self._calc_perplexity(n_idx, " ".join(words_best))

        return words_best, perplexity_best

    def ILS_kick(
        self, n_idx: int, words: list[str], n_kick: int = 2
    ) -> tuple[list[str], list[int]]:
        words = words.copy()
        neighbor_types = []
        if n_kick == 2:
            length = 10
            left = random.randint(0, len(words) - length)
            right = left + length
            removed = words[left:right]
            words = words[:left] + words[right:]
            neighbor_type = [left]
            for word in removed:
                insert_idx = random.randint(0, len(words))
                words.insert(insert_idx, word)
                neighbor_type.append(insert_idx)
            neighbor_types.append(tuple(neighbor_type))

        strength = [2, 3, 3, 4, 5, 10]
        for _ in range(n_kick * strength[n_idx]):
            r0 = random.randint(0, len(words) - 1)
            r1 = random.randint(0, len(words) - 1)
            words[r0], words[r1] = words[r1], words[r0]
            neighbor_types.append((r0, r1))
        return words, neighbor_types

    def run(self, list_idx_target: Optional[list[int]] = None):
        if list_idx_target is None:
            list_idx_target = list(range(NUM_PROBLEMS))
        list_population_size = [4] * NUM_PROBLEMS
        for n_idx in itertools.cycle(list_idx_target):
            free_memory()
            words_best, perplexity_best_old = self._get_best(n_idx)
            print("#" * 80)
            print(f"[run] n_idx:{n_idx} perplexity_best:{perplexity_best_old:.2f}")
            # words_best, perplexity_best = self._hillclimbing(
            #     n_idx,
            #     words_best,
            #     perplexity_best_old,
            #     score_estimator=self.score_estimators[n_idx],
            #     iter_total=500,
            # )
            population_size = list_population_size[n_idx]
            words_best, perplexity_best = self._beam_search(
                n_idx,
                self.score_estimators[n_idx],
                population_size=population_size,
                iter_total=1000,
            )
            print(f"[run] n_idx:{n_idx} perplexity_best:{perplexity_best:.2f}")
            did_kick = False
            if perplexity_best_old == perplexity_best:
                if words_best == self._get_best_all(n_idx)[0]:
                    self.list_num_kick[n_idx] = 0
                # reset + 4 -> 3 -> 2 -> 1 -> reset + 4 -> 3 -> 2 -> 1 -> ...
                self.list_num_kick[n_idx] -= 1
                flag_reset = self.list_num_kick[n_idx] <= 0
                if flag_reset:
                    self.list_num_kick[n_idx] = random.randint(2, 3)
                n_kick = self.list_num_kick[n_idx]
                did_kick = True
                if flag_reset:
                    print("[run] Reset words")
                    words_best = self._get_best_all(n_idx)[0]
                words_best, neighbor_types = self.ILS_kick(
                    n_idx, words_best, n_kick=n_kick
                )
                print(f"[run] Apply {n_kick} kicks: {neighbor_types}")
                perplexity_best = self._calc_perplexity(n_idx, " ".join(words_best))
            self.list_words_best[n_idx] = words_best
            self.list_perplexity_best[n_idx] = perplexity_best
            self._update_best_all(n_idx, words_best, perplexity_best)
            if not did_kick and perplexity_best < self._get_best_all(n_idx)[1] * 1.1:
                save_text(self._calc_perplexity, n_idx, " ".join(words_best), verbose=1)
            if time() > self.last_time_score_memo_saved + 1800:
                save_score_memo(self.score_memo, self.score_memo_with_error)
                self.last_time_score_memo_saved = time()
                self.score_estimators[n_idx].save_model()
            list_population_size[n_idx] *= 2

if __name__ == "__main__":
    optimizer = Optimization(flag_use_best=False)
    optimizer.run()
