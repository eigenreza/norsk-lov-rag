"""A small lexical search, used to check that unanswerable questions are not answered by the corpus.

It is plain BM25 over paragraphs, with the Norwegian Snowball stemmer, so that forms such as
"personfradraget" and "personfradrag" match. The search only needs to find candidate texts; a
model then decides whether any of them answers the question.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache

import snowballstemmer

# Part of the key of remembered search hits, so that a changed search is run again.
SEARCH_VERSION = "bm25-snowball-v1"
TOKEN = re.compile(r"[0-9a-zæøåéü]+")
STOP_WORDS = frozenset(
    """og i på til av for som en et ei er det de den med at om ikke kan skal har eller fra ved
    etter når hvis også jf eg jeg du man hva hvor hvordan hvilken hvilke kva kor korleis ein eit
    ikkje frå meg min mitt mine seg sin sitt sine vi oss år""".split()
)
_STEMMER = snowballstemmer.stemmer("norwegian")


@lru_cache(maxsize=500_000)
def stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def tokens(text: str) -> list[str]:
    return [
        stem(word)
        for word in TOKEN.findall(text.lower())
        if word not in STOP_WORDS and len(word) > 1
    ]


class BM25:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.ids = ids
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for index, text in enumerate(texts):
            counts = Counter(tokens(text))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((index, count))
        self.average = sum(self.lengths) / max(1, len(self.lengths))

    def search(self, query: str, k: int = 6) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        size = len(self.ids)
        for term in set(tokens(query)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            idf = math.log(1 + (size - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, count in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average)
                scores[index] += idf * count * (self.k1 + 1) / (count + norm)
        best = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:k]
        return [(self.ids[index], score) for index, score in best]
