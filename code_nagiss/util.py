import hashlib
import pickle
import warnings
from typing import Union

import numpy as np
from constants import DF_INPUT, LIST_NUM_WORDS, LIST_WORD_TO_ID, NUM_PROBLEMS, PATH_SAVE
from tqdm.auto import tqdm


def get_path_words_best(n_idx):
    path_save_idx = PATH_SAVE / f"{n_idx:04d}"
    words_original = DF_INPUT.loc[n_idx, "text"].split(" ")

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
    path_save_idx = PATH_SAVE / f"{n_idx:04d}"
    if not path_save_idx.exists():
        path_save_idx.mkdir()
    text_original = DF_INPUT.loc[n_idx, "text"]
    words_original = text_original.split(" ")
    words = text.split(" ")
    if sorted(words) != sorted(words_original):
        print(f"words are different: {words} != {words_original}")
        return
    text = " ".join(words)
    score = get_perplexity(n_idx, text)
    if verbose >= 1:
        print(f"score:{score:.4f}")
    if verbose >= 2:
        print(text)
    md5 = hashlib.md5(text.encode()).hexdigest()
    path_save_text = path_save_idx / f"{score:.4f}_{md5}.txt"

    with path_save_text.open("w") as f:
        f.write(text)

    return score


def compress_words(problem_id: int, words: list[str]) -> bytes:
    assert len(words) == LIST_NUM_WORDS[problem_id]
    word_to_id = LIST_WORD_TO_ID[problem_id]
    return bytes([word_to_id[word] for word in words])


def compress_text(problem_id: int, text: str) -> bytes:
    return compress_words(problem_id, text.split())


def compress_score_memo_file(name: str):
    path_compressed_score_memo = PATH_SAVE / name
    path_legacy_score_memo = path_compressed_score_memo.with_suffix(".pkl")
    with path_legacy_score_memo.open("rb") as f:
        legacy_score_memo: dict[str, float] = pickle.load(f)

    list_compressed_score_memo: list[dict[bytes, float]] = [
        {} for _ in range(NUM_PROBLEMS)
    ]
    print(f"[compress_score_memo_file] compressing {name}...")
    for text, score in tqdm(legacy_score_memo.items(), mininterval=30):
        words = text.split()
        length = len(words)
        problem_id = -1
        if length == 10:
            problem_id = 0
        elif length == 20:
            if "elf" in words:
                problem_id = 1
            elif "nice" in words:
                problem_id = 2
        elif length == 30:
            problem_id = 3
        elif length == 50:
            problem_id = 4
        elif length == 100:
            problem_id = 5
        if problem_id == -1:
            warnings.warn(f"problem_id cannot be identified: {text}")
            continue
        list_compressed_score_memo[problem_id][
            compress_words(problem_id, words)
        ] = score

    path_compressed_score_memo.mkdir(parents=True, exist_ok=True)
    for problem_id, compressed_score_memo in enumerate(list_compressed_score_memo):
        with (path_compressed_score_memo / f"{problem_id:04d}.pkl").open("wb") as f:
            pickle.dump(compressed_score_memo, f)


def load_score_memo() -> tuple[list[dict[bytes, float]], list[dict[bytes, float]]]:
    def load(name: str) -> list[dict[bytes, float]]:
        path_score_memo = PATH_SAVE / name
        if not path_score_memo.exists():
            if path_score_memo.with_suffix(".pkl").exists():
                compress_score_memo_file(name)
            else:
                return [{} for _ in range(NUM_PROBLEMS)]
        assert path_score_memo.is_dir()
        list_score_memo = []
        for problem_id in range(NUM_PROBLEMS):
            path_problem_score_memo = path_score_memo / f"{problem_id:04d}.pkl"
            if path_problem_score_memo.exists():
                with path_problem_score_memo.open("rb") as f:
                    score_memo = pickle.load(f)
                list_score_memo.append(score_memo)
            else:
                warnings.warn(f"{path_problem_score_memo} does not exist")
                list_score_memo.append({})
        return list_score_memo

    return load("score_memo"), load("score_memo_with_error")


def save_score_memo(
    score_memo: list[dict[bytes, float]],
    score_memo_with_error: list[dict[bytes, float]],
):
    def save(name: str, score_memo: list[dict[bytes, float]]):
        path_score_memo = PATH_SAVE / name
        path_score_memo.mkdir(parents=True, exist_ok=True)
        for problem_id, problem_score_memo in enumerate(score_memo):
            with (path_score_memo / f"{problem_id:04d}.pkl").open("wb") as f:
                pickle.dump(problem_score_memo, f)

    score_memo_original, score_memo_with_error_original = load_score_memo()
    for original, arg in zip(score_memo_original, score_memo):
        original.update(arg)
    for original, arg in zip(score_memo_with_error_original, score_memo_with_error):
        original.update(arg)
    save("score_memo", score_memo_original)
    save("score_memo_with_error", score_memo_with_error_original)


def get_perplexity_(
    scorer,
    problem_id: int,
    score_memo: list[dict[bytes, float]],
    score_memo_with_error: list[dict[bytes, float]],
    text: Union[str, list[str]],
) -> Union[float, list[float]]:
    problem_score_memo = score_memo[problem_id]
    problem_score_memo_with_error = score_memo_with_error[problem_id]
    if isinstance(text, str):
        compressed_text = compress_text(problem_id, text)
        if compressed_text in problem_score_memo:
            return problem_score_memo[compressed_text]
        score: float = scorer.get_perplexity(text)
        problem_score_memo[compressed_text] = score
        return score
    elif isinstance(text, list):
        list_text = text
        list_compressed_text: list[bytes] = []
        list_text_new: list[str] = []
        list_compressed_text_new: list[bytes] = []
        for text in list_text:
            compressed_text = compress_text(problem_id, text)
            list_compressed_text.append(compressed_text)
            if (
                compressed_text not in problem_score_memo
                and compressed_text not in problem_score_memo_with_error
            ):
                list_text_new.append(text)
                list_compressed_text_new.append(compressed_text)

        if len(list_text_new):
            list_score_new: list[float] = scorer.get_perplexity(list_text_new)
            if len(list_text_new) == 1:
                problem_score_memo[list_compressed_text_new[0]] = list_score_new[0]
            else:
                for compressed_text, score in zip(
                    list_compressed_text_new, list_score_new
                ):
                    problem_score_memo_with_error[compressed_text] = score
        else:
            list_score_new = []

        list_score = []
        for compressed_text in list_compressed_text:
            if compressed_text in problem_score_memo:
                list_score.append(problem_score_memo[compressed_text])
            elif compressed_text in problem_score_memo_with_error:
                list_score.append(problem_score_memo_with_error[compressed_text])
            else:
                assert False

        return list_score

    else:
        raise TypeError("text should be str or list[str]")
