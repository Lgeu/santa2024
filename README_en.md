# Santa 2024 Hamiltonians' solution

## Overview

We employ an Iterated Local Search (ILS) framework for our solution. In broad terms, the process is as follows:

1. Perform a local (greedy) search on the current solution (sequence of words) to improve it.
2. Once no further improvement is possible, use a "kick" (perturbation) to escape from the local optimum.

We repeat these two steps in a loop.

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

Within `kick()`, we carry out operations like:

- Randomly select 2 words from the sequence and swap their positions.
- Randomly select a continuous segment of the sequence, remove it, then reinsert each removed word at random positions.

## Local Search

```python
def depth_first_search(best_score, words, depth):
    for neighbor_words in make_neighbors(words):
        if score(neighbor_words) < score(words):
            # If there's an improvement, stop and return
            return neighbor_words
        elif score(neighbor_words) < get_threshold(best_score, depth):
            # If a neighbor seems promising, explore around it as well
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

For local search, we perform a depth-first search as shown above, continuously improving the solution as long as it leads to a lower score.

- `make_neighbors()` generates all sequences obtained by taking a small number of words out of the sequence and inserting them elsewhere.
- `get_threshold()` returns a value like `best_score * 1.1` when the depth is small and `best_score` when the depth is large.

In the actual implementation, we also include:
- Early termination based on node count limits
- More efficient batch score calculation
- Pruning using a lightweight Neural Network (NN)

## NN-Based Pruning

Exact score computation is very costly, so we first use a lightweight NN to screen for promising neighbors. This allows us to reduce the number of sequences for which we compute the exact score (using Gemma) to below 1/10 of the total.

The NN has around 1.3 million parameters (less than 1/5000 the size of Gemma, which is 9B parameters) and is optimized online during the search, in a manner similar to reinforcement learning.
