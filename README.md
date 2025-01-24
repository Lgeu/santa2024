# Santa 2024 Hamiltonians' solution

## 概要

解法の大きい枠組みとして反復局所探索法 (ILS) を使用しており、大まかに言えば

1. 現在の解 (単語列) を局所的に探索して貪欲に改善を図る
2. 改善できなくなったら kick (perturbation) によって局所解から抜け出す

という流れを繰り返します。

```python
def solve(initial_words):
    best_words = initial_words
    while True:
        words = best_words
        for _ in range(3):
            words = kick(words)
            words = local_search(words)
            if score(words) < score(best_words):
                best_words = words
```

`kick()` 内では、

* 単語列からランダムに 2 単語を選び位置を入れ替える
* 単語列から連続した区間をランダムに選んで消去し、消された単語をそれぞれランダムな箇所に挿入する

のような操作を行います。

## 近傍探索

```python
def depth_first_search(best_score, words, depth):
    for neighbor_words in make_neighbors(words):
        if score(neighbor_words) < score(initial_words):
            # 改善すれば終了
            return neighbor_words
        elif score(neighbor_words) < get_threshold(best_score, depth):
            # 改善しそうな単語列があればその周辺も探索
            result = depth_first_search(best_score, neighbor_words, depth + 1)
            if result is not None:
                return result

def local_search(initial_words):
    words = initial_words
    while True:
        result = depth_first_search(score(words), words, 0)
        if result is None:
            return words
        words = result
```

近傍探索では、上記のような深さ優先探索を行って、スコアが減る限り解を改善し続けます。

`make_neighbors()` は、単語列から少数個の単語を抜き出して別の箇所に挿入した単語列の集合を生成します。
`get_threshold()` は、depth が小さい時には `best_score * 1.1` などの値を、depth が大きい時には `best_score` を返します。

実際のコードでは、

* ノード数の制限による探索打ち切り
* スコアのバッチ計算による効率化
* NN による枝刈り

なども行っています。

## NN による枝刈り

正確なスコア計算に要する計算量が大きすぎるため、事前に軽量な NN で有望な近傍を絞り込み、Gemma を使って正確にスコアを計算する単語列の数を 1/10 以下に減らしています。

NN は 1.3M 程度 (Gemma 9B の 1/5000 以下) のパラメータを持ち、強化学習のように探索中にオンラインで最適化します。
