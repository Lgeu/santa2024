import argparse
import math
import subprocess
import time
from pathlib import Path
from typing import Optional

import pandas as pd


def get_best_txt_file(
    n_idx: int, words_original: list[str], path_save: Path
) -> tuple[Optional[float], Optional[str]]:
    """問題番号を指定して、最小スコアのテキストファイルを取得する。"""
    path_save_idx = path_save / f"{n_idx:04d}"
    list_path_txt = list(path_save_idx.glob("*.txt"))
    if not list_path_txt:
        print(f"No txt files in {path_save_idx}")
        return None, None
    list_scores_paths: list[tuple[float, Path]] = []
    for path in list_path_txt:
        try:
            score = float(path.stem.split("_")[0])
            list_scores_paths.append((score, path))
        except ValueError:
            print(f"Invalid filename {path}, cannot convert to float.")
            continue
    if not list_scores_paths:
        print(f"No valid score files in {path_save_idx}")
        return None, None
    # Find the path with the minimum score
    score_min, path_min = min(list_scores_paths, key=lambda x: x[0])
    text_min = path_min.read_text()
    words_min = text_min.split()
    if sorted(words_min) != sorted(words_original):
        print(f"Words mismatch at index {n_idx}")
        return None, None
    return score_min, text_min


def update_submission(
    df: pd.DataFrame, path_save: Path, path_save_submissions: Path
) -> tuple[Optional[Path], Optional[list[float]]]:
    """指定されたパスに存在する解のテキストファイルから提出用 CSV を作成する。"""
    df_submission = df.copy()
    list_scores_submission = []
    for n_idx, row in df.iterrows():
        text_original = row["text"]
        words_original = text_original.split()
        score_min, text_min = get_best_txt_file(n_idx, words_original, path_save)
        if score_min is None:
            list_scores_submission.append(math.inf)
            continue
        df_submission.at[n_idx, "text"] = text_min
        list_scores_submission.append(score_min)
    assert len(list_scores_submission) == len(df)
    if all(not math.isfinite(score) for score in list_scores_submission):
        print("No scores found, cannot create submission.")
        return None, None
    score_ave = sum(list_scores_submission) / len(df)
    print(f"Average score: {score_ave}")
    path_save_submission = path_save_submissions / f"submission_{score_ave:.6f}.csv"
    df_submission.to_csv(path_save_submission, index=False)
    return path_save_submission, list_scores_submission


def get_best_score_from_submissions(
    path_save_submissions: Path,
) -> tuple[Optional[Path], Optional[float]]:
    """スコアが最小の提出ファイルを取得する。"""
    list_path_csv = list(path_save_submissions.glob("submission_*.csv"))
    if not list_path_csv:
        return None, None
    list_scores_paths = []
    for path in list_path_csv:
        try:
            # Extract score from filename
            score_str = path.stem.split("_")[-1]
            score = float(score_str)
            list_scores_paths.append((score, path))
        except ValueError:
            print(f"Invalid filename {path}, cannot extract score.")
            continue
    if not list_scores_paths:
        return None, None
    score_best, path_best = min(list_scores_paths, key=lambda x: x[0])
    print(f"Best score: {score_best}")
    print(f"Best submission path: {path_best}")
    return path_best, score_best


def submit_to_kaggle(path_submission, comment="submit"):
    cmd = (
        f"kaggle competitions submit -c santa-2024 -f {path_submission} -m '{comment}'"
    )
    print(f"Submitting to Kaggle: {cmd}")
    subprocess.run(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(
        description="Update and submit Kaggle submissions."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force submit the new submission."
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run in a loop, submitting new best submissions.",
    )
    args = parser.parse_args()

    path_input_csv = Path("../input/santa-2024/sample_submission.csv")
    path_save = Path("./save")
    path_save_submissions = path_save / "submissions"
    path_save_submissions.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path_input_csv)

    if args.force:
        path_submission, scores = update_submission(
            df, path_save, path_save_submissions
        )
        if path_submission:
            print(f"Force submitting: {path_submission}")
            scores_str = ", ".join(f"{score}" for score in scores)
            print(f"Scores: {scores_str}")
            submit_to_kaggle(path_submission, comment=scores_str)
    elif args.loop:
        while True:
            _, score_old = get_best_score_from_submissions(path_save_submissions)
            score_old = score_old or math.inf
            path_submission, scores = update_submission(
                df, path_save, path_save_submissions
            )
            path_best_new, score_new = get_best_score_from_submissions(
                path_save_submissions
            )
            score_new = score_new or math.inf
            if score_new < score_old:
                print("New best submission found.")
                print(f"Best score improved: {score_old} -> {score_new}")
                scores_str = ", ".join(f"{score}" for score in scores)
                print(f"Scores: {scores_str}")
                submit_to_kaggle(path_best_new, comment=scores_str)
            else:
                print("No improvement in score.")
            time.sleep(60 * 30)
    else:
        path_submission, scores = update_submission(
            df, path_save, path_save_submissions
        )
        if path_submission:
            print(f"Submission created at: {path_submission}")
            print(f"Scores: {', '.join(f'{score}' for score in scores)}")
        else:
            print("Submission could not be created.")


if __name__ == "__main__":
    main()
