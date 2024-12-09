import copy
import gc
import itertools
import random
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
            raise FileNotFoundError

        self.word_to_id = LIST_WORD_TO_ID[problem_id]

        self.device = device
        self.model = SantaNet(
            vocab_size=len(self.word_to_id), channels=128, num_blocks=12
        ).to(device)
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
        return preds.detach().cpu().numpy()

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

    for length in range(1, 3):
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
        self.score_estimators = {
            0: None,
            1: None,
            2: None,
            3: None,
            4: None,
            5: ScoreEstimator(
                problem_id=5,
                epoch=11,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            ),
        }

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
            depth_to_threshold = {
                0: 1.01,
                1: 1.007,
                2: 1.005,
                3: 1.004,
                4: 1.003,
                5: 1.0025,
                6: 1.002,
                7: 1.0015,
                8: 1.001,
                9: 1.001,
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

            neighbors = make_neighbors(words)
            max_depth = depth
            for _ in itertools.count(0):
                list_words_nxt: list[list[str]] = []
                list_texts_nxt: list[str] = []
                list_neighbor_type: list = []

                num_candidates = 1024
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
                if len(list_words_nxt) < num_candidates:
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
                        f" nxt:{perplexity_nxt:.2f}"
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

    def ILS_kick(
        self, words: list[str], n_kick: int = 2
    ) -> tuple[list[str], list[int]]:
        words = words.copy()
        neighbor_types = []
        for _ in range(min(20, n_kick * 3 + 3)):
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
            words_best, perplexity_best_old = self._get_best(n_idx)
            print("#" * 80)
            print(f"[run] n_idx:{n_idx} perplexity_best:{perplexity_best_old:.2f}")
            words_best, perplexity_best = self._hillclimbing(
                n_idx,
                words_best,
                perplexity_best_old,
                score_estimator=self.score_estimators[n_idx],
                iter_total=500,
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


if __name__ == "__main__":
    optimizer = Optimization()
    optimizer.run([5])
