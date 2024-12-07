from pathlib import Path

import pandas as pd

PATH_GEMMA = Path("../input/gemma-2/")
PATH_INPUT_CSV = Path("../input/santa-2024/sample_submission.csv")
DF_INPUT = pd.read_csv(PATH_INPUT_CSV)

PATH_SAVE = Path("./save")
PATH_SAVE.mkdir(parents=True, exist_ok=True)

NUM_PROBLEMS = len(DF_INPUT)
assert NUM_PROBLEMS == 6

LIST_NUM_WORDS = [
    len(DF_INPUT.loc[problem_id, "text"].split()) for problem_id in range(NUM_PROBLEMS)
]
assert LIST_NUM_WORDS == [10, 20, 20, 30, 50, 100]

LIST_WORD_TO_ID: list[dict[str, int]] = [
    {
        word: i
        for i, word in enumerate(sorted(set(DF_INPUT.loc[problem_id, "text"].split())))
    }
    for problem_id in range(NUM_PROBLEMS)
]
# fmt: off
assert LIST_WORD_TO_ID[5] == {'advent': 0, 'and': 1, 'angel': 2, 'as': 3, 'bake': 4, 'beard': 5, 'believe': 6, 'bow': 7, 'candle': 8, 'candy': 9, 'card': 10, 'carol': 11, 'cheer': 12, 'chimney': 13, 'chocolate': 14, 'cookie': 15, 'decorations': 16, 'doll': 17, 'dream': 18, 'drive': 19, 'eat': 20, 'eggnog': 21, 'elf': 22, 'family': 23, 'fireplace': 24, 'from': 25, 'fruitcake': 26, 'game': 27, 'gifts': 28, 'gingerbread': 29, 'give': 30, 'greeting': 31, 'grinch': 32, 'have': 33, 'hohoho': 34, 'holiday': 35, 'holly': 36, 'hope': 37, 'in': 38, 'is': 39, 'it': 40, 'jingle': 41, 'joy': 42, 'jump': 43, 'kaggle': 44, 'laugh': 45, 'magi': 46, 'merry': 47, 'milk': 48, 'mistletoe': 49, 'naughty': 50, 'nice': 51, 'night': 52, 'not': 53, 'nutcracker': 54, 'of': 55, 'ornament': 56, 'paper': 57, 'peace': 58, 'peppermint': 59, 'poinsettia': 60, 'polar': 61, 'puzzle': 62, 'reindeer': 63, 'relax': 64, 'scrooge': 65, 'season': 66, 'sing': 67, 'sleep': 68, 'sleigh': 69, 'snowglobe': 70, 'star': 71, 'stocking': 72, 'that': 73, 'the': 74, 'to': 75, 'toy': 76, 'unwrap': 77, 'visit': 78, 'walk': 79, 'we': 80, 'wish': 81, 'with': 82, 'wonder': 83, 'workshop': 84, 'wrapping': 85, 'wreath': 86, 'you': 87, 'yuletide': 88}
# fmt: on
