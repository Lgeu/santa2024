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
        n_pop: int=16,
        flag_use_best=True,  # best を使うかどうか
        flag_shuffle=True,  # best を使わない時にシャッフルするかどうか
    ):
        self.n_pop = n_pop
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
        self.list_words_pop: list[list[list[str]]] = []
        self.list_perplexity_pop: list[list[float]] = []
        self.list_edge_count_pop: list[list[list[int]]] = []
        self.list_depth_pop: list[list[int]] = []
        self.no_update_count: list[int] = [0] * NUM_PROBLEMS
        self.list_visited: list[set[tuple[str]]] = [set() for _ in range(NUM_PROBLEMS)]
        self.list_perplexity_curr_best: list[float] = []
        self.word2idx: list[dict[str, int]] = []
        for idx in range(NUM_PROBLEMS):
            words_pop = []
            perplexity_pop = []
            if self.flag_use_best: # TODO: load population
                _, list_words = get_path_words_best(idx)
                assert list_words is not None
            else:
                text: str = DF_INPUT.iloc[idx, 1]
                list_words = text.split()
                if self.flag_shuffle:
                    random.shuffle(list_words)
            text = " ".join(list_words)
            score_new = self._calc_perplexity(idx, text)
            for _ in range(self.n_pop):
                words_pop.append(list_words.copy())
                perplexity_pop.append(score_new)
            all_words = ["<bos>"] + list(set(list_words)) # dummy word eos = bos
            word2idx = {w: i for i, w in enumerate(all_words)}
            self.word2idx.append(word2idx)
            n_words = len(all_words)
            edge_count_pop = [[0] * n_words for _ in range(n_words)]
            for i in range(self.n_pop):
                edge_count_pop[0][word2idx[list_words[0]]] += 1 # bos -> first word
                edge_count_pop[word2idx[list_words[-1]]][0] += 1 # last word -> eos
                for j in range(n_words - 2): # -1 は bos
                    left_word = list_words[j]
                    right_word = list_words[j + 1]
                    edge_count_pop[word2idx[left_word]][word2idx[right_word]] += 1
            self.list_words_pop.append(words_pop)
            self.list_perplexity_pop.append(perplexity_pop)
            self.list_edge_count_pop.append(edge_count_pop)
            self.list_depth_pop.append([0] * self.n_pop)
            self.list_perplexity_curr_best.append(np.min(perplexity_pop))
            print(f"idx:{idx} score:{score_new:.4f}")

        # 行き詰まった時に戻るためのガチの現在までの最良の解
        self.list_words_pop_best = copy.deepcopy(self.list_words_pop)
        self.list_perplexity_pop_best = copy.deepcopy(self.list_perplexity_pop)

        # 初期化
        self.list_num_kick = [1] * NUM_PROBLEMS

    def _calc_perplexity(
        self, n_idx: int, text: Union[str, list[str]]
    ) -> Union[float, list[float]]:
        return get_perplexity_(
            self.calculator, n_idx, self.score_memo, self.score_memo_with_error, text
        )

    def _get_pop(self, n_idx: int) -> tuple[list[list[str]], list[float]]:
        return self.list_words_pop[n_idx], self.list_perplexity_pop[n_idx]

    def _update_best_all(self, n_idx: int, words: list[str], perplexity: float):
        if perplexity < self.list_perplexity_best_all[n_idx]:
            self.list_words_best_all[n_idx] = words.copy()
            self.list_perplexity_best_all[n_idx] = perplexity

    def _get_best_all(self, n_idx: int) -> tuple[list[list[str]], list[float]]:
        return self.list_words_pop_best[n_idx], self.list_perplexity_pop_best[n_idx]

    def _print_pop(self, n_idx: int):
        words_pop = self.list_words_pop[n_idx]
        perplexity_pop = self.list_perplexity_pop[n_idx]
        depth_pop = self.list_depth_pop[n_idx]
        print(f"idx:{n_idx} depth:{depth_pop}")
        for words, ppl, depth in zip(words_pop, perplexity_pop, depth_pop):
            print(f"  {ppl:.2f} {' '.join(words)}")

    def _get_edge_entropy(self, edge_count_pop: list[list[int]]) -> float:
        h = 0.0
        n_words = len(edge_count_pop)
        for i in range(n_words):
            for j in range(n_words):
                p = edge_count_pop[i][j] / self.n_pop
                if p > 0:
                    h += p * math.log(p)
        return -h

    def _add_edge_count(self, edge_count_pop: list[list[int]], words: list[str], value: int, word2idx: dict[str, int]):
        n_words = len(words)
        edge_count_pop[0][word2idx[words[0]]] += value # bos -> first word
        edge_count_pop[word2idx[words[-1]]][0] += value # last word -> eos
        for j in range(n_words - 1):
            left_word = words[j]
            right_word = words[j + 1]
            edge_count_pop[word2idx[left_word]][word2idx[right_word]] += value
        return edge_count_pop

    def _eval_improvement(self, old_words, new_words, old_score, new_score, old_edge_count_pop, word2idx):
        old_edge_entropy = self._get_edge_entropy(old_edge_count_pop)
        new_edge_count_pop = self._add_edge_count(copy.deepcopy(old_edge_count_pop), old_words, -1, word2idx)
        new_edge_count_pop = self._add_edge_count(new_edge_count_pop, new_words, 1, word2idx)
        new_edge_entropy = self._get_edge_entropy(new_edge_count_pop)
        score_diff = new_score - old_score
        edge_entropy_diff = new_edge_entropy - old_edge_entropy 
        if score_diff > 0:
            return 0, score_diff, edge_entropy_diff
        elif edge_entropy_diff < 0:
            return score_diff / edge_entropy_diff, score_diff, edge_entropy_diff
        else:
            return -score_diff * 10000, score_diff, edge_entropy_diff

    def _beam_search(
        self,
        n_idx: int,
        score_estimator: ScoreEstimator,
    ) -> tuple[list[str], float]:

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
        word2idx = self.word2idx[n_idx]
        words_pop = self.list_words_pop[n_idx]
        perplexity_pop = self.list_perplexity_pop[n_idx]
        edge_count = self.list_edge_count_pop[n_idx]
        depth_pop = self.list_depth_pop[n_idx]

        perplexity_best = np.min(perplexity_pop)
        words_best = words_pop[np.argmin(perplexity_pop)]

        visited = self.list_visited[n_idx]
        for words in words_pop:
            visited.add(tuple(words))

        def search(
            words_idx: int, depth: int = 0,
        ) -> tuple[float, list[str], list[int]]:
            words = words_pop[words_idx]
            ppl = perplexity_pop[words_idx]
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
                    return None, None, None, depth, None, None, None

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

                improvements = []
                score_diffs = [] # debug
                entropy_diffs = [] # debug
                for ppl_nxt_with_error, words_nxt in zip(list_perplexity_nxt_with_error, list_words_nxt):
                    eval_improvement, score_diff, entropy_diff = self._eval_improvement(words, words_nxt, ppl, ppl_nxt_with_error, edge_count, word2idx)
                    improvements.append(eval_improvement)
                    score_diffs.append(score_diff)
                    entropy_diffs.append(entropy_diff)

                estimated_rank = int(np.argmax(improvements))
                words_nxt = list_words_nxt[estimated_rank]
                perplexity_nxt_with_error = list_perplexity_nxt_with_error[
                    estimated_rank
                ]
                neighbor_type = list_neighbor_type[estimated_rank]
                improvement = improvements[estimated_rank]
                score_diff = score_diffs[estimated_rank]
                entropy_diff = entropy_diffs[estimated_rank]

                if perplexity_nxt_with_error < perplexity_best + 2.0:
                    perplexity_nxt = self._calc_perplexity(n_idx, " ".join(words_nxt))
                else:
                    perplexity_nxt = perplexity_nxt_with_error

                if perplexity_nxt < ppl:
                    stats.accepted[estimated_rank] += 1
                    return perplexity_nxt, words_nxt, [neighbor_type], 0, improvement, score_diff, entropy_diff
                elif perplexity_nxt < ppl * depth_to_threshold[depth]:
                    stats.accepted[estimated_rank] += 1
                    return perplexity_nxt, words_nxt, [neighbor_type], depth + 1, improvement, score_diff, entropy_diff
                else:
                    stats.rejected[estimated_rank] += 1
                    return None, None, None, depth, None, None, None

                #     print(
                #         f"[hillclimbing] iter:{pbar.n} best:{perplexity_best:.2f}"
                #         f" nxt:{perplexity_nxt or math.inf:.2f}"
                #         f" neighbor:{neighbor_type}"
                #         f" depth:{depth}"
                #         f" {stats.summary()}"
                #     )
        for i in range(self.n_pop):
            perplexity_nxt, words_nxt, neighbor_types, depth_nxt, improvement, score_diff, entropy_diff = search(i, depth_pop[i])
            if perplexity_nxt is not None:
                edge_count = self._add_edge_count(edge_count, words_pop[i], -1, word2idx)
                edge_count = self._add_edge_count(edge_count, words_nxt, 1, word2idx)
                words_pop[i] = words_nxt
                perplexity_pop[i] = perplexity_nxt
                depth_pop[i] = depth_nxt

                # assert perplexity_nxt < perplexity_best
        #     print(
        #         f"[hillclimbing] Update: {perplexity_best:.2f}"
        #         f" -> {perplexity_nxt:.2f},"
        #         f" neighbor:{','.join(map(str, neighbor_types))}"
        #         f" max_depth:{max_depth}"
        #         f" {stats.summary()}"
        #     )
        #     perplexity_best = perplexity_nxt
        #     words_best = words_nxt
        # else:
        #     print(f"[hillclimbing] No update, max_depth:{max_depth} {stats.summary()}")

        best_idx = np.argmin(perplexity_pop)
        words_best_new = words_pop[best_idx]
        perplexity_best_new = perplexity_pop[best_idx]
        if perplexity_best_new < perplexity_best:
            print(
                f"[beam_search] Update: {perplexity_best:.2f}"
                f" -> {perplexity_best_new:.2f},"
                f" {stats.summary()}"
            )
            words_best = words_best_new
            perplexity_best = perplexity_best_new
        
        self.list_words_pop[n_idx] = words_pop
        self.list_perplexity_pop[n_idx] = perplexity_pop
        self.list_edge_count_pop[n_idx] = edge_count
        self.list_depth_pop[n_idx] = depth_pop
        self.list_visited[n_idx] = visited

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
        for n_idx in itertools.cycle(list_idx_target):
            free_memory()
            words_pop, perplexity_pop = self._get_pop(n_idx)
            best_idx = np.argmin(perplexity_pop)
            words_best = words_pop[best_idx]
            perplexity_best_old = perplexity_pop[best_idx]

            print("#" * 80)
            print(f"[run] n_idx:{n_idx} perplexity_best:{perplexity_best_old:.2f}")
            words_best, perplexity_best = self._beam_search(
                n_idx,
                score_estimator=self.score_estimators[n_idx],
            )
            print(f"[run] n_idx:{n_idx} perplexity_best:{perplexity_best:.2f}")
            did_kick = False
            if perplexity_best < self.list_perplexity_curr_best[n_idx]:
                self.no_update_count[n_idx] = 0
                self.list_perplexity_curr_best[n_idx] = perplexity_best
            else:
                self.no_update_count[n_idx] += 1

            if self.no_update_count[n_idx] > 10:
                self.no_update_count[n_idx] = 0
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
                    self.list_words_pop[n_idx] = self._get_best_all(n_idx)[0]
                for i in range(self.n_pop):
                    self.list_words_pop[n_idx][i], neighbor_types = self.ILS_kick(
                        n_idx, self.list_words_pop[n_idx][i], n_kick=n_kick
                    )
                # print(f"[run] Apply {n_kick} kicks: {neighbor_types}")
                print(f"[run] Apply {n_kick} kicks for each population")
                for i in range(self.n_pop): # update perplexity
                    self.list_perplexity_pop[n_idx][i] = self._calc_perplexity(n_idx, " ".join(self.list_words_pop[n_idx][i]))
                    self.list_depth_pop[n_idx][i] = 0
                self.list_perplexity_curr_best[n_idx] = np.min(self.list_perplexity_pop[n_idx])
                self.list_visited[n_idx] = set()
            # self._update_best_all(n_idx, words_best, perplexity_best)
            if not did_kick and perplexity_best < np.min(self._get_best_all(n_idx)[1]) * 1.1:
                save_text(self._calc_perplexity, n_idx, " ".join(words_best), verbose=1)

            self._print_pop(n_idx) # debug

            if time() > self.last_time_score_memo_saved + 1800:
                save_score_memo(self.score_memo, self.score_memo_with_error)
                self.last_time_score_memo_saved = time()
                self.score_estimators[n_idx].save_model()

# %%
if __name__ == "__main__":
    # GPU warm up
    print("Warming up GPU...")
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    ).cuda()
    dummy_input = torch.randn(100, 10).cuda()
    with torch.no_grad():
        _ = dummy_model(dummy_input)
    print("GPU warm up done")

    # %%
    optimizer = Optimization()
    # %%
    optimizer.run([5])
    # %%

