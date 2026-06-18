"""BM25 with Norwegian handling: lower case, stop words, the Snowball stemmer and, as an option,
simple compound splitting.

Norwegian writes compounds as one word, so "merverdiavgiftsregisteret" shares no token with
"registeret" or "merverdiavgift". The splitter looks for two known words that make up a long
word, allowing the linking letters s and e between them, and adds the parts as extra tokens.
"Known" means seen often enough in the corpus itself, so no outside word list is needed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from norsk_lov_rag.testset.search import STOP_WORDS, TOKEN, stem

MIN_WORD = 9
MIN_PART = 4
MIN_COUNT = 5
LINKS = "se"


class CompoundSplitter:
    def __init__(self, counts: Counter[str]):
        self.counts = counts
        self._cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_texts(cls, texts) -> CompoundSplitter:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(TOKEN.findall(text.lower()))
        return cls(counts)

    def _known(self, word: str) -> bool:
        return len(word) >= MIN_PART and self.counts.get(word, 0) >= MIN_COUNT

    def split(self, word: str) -> tuple[str, ...]:
        """The parts of a compound, or an empty tuple. The right part may be split again."""
        if word in self._cache:
            return self._cache[word]
        best: tuple[int, str, str] | None = None
        if len(word) >= MIN_WORD:
            for cut in range(MIN_PART, len(word) - MIN_PART + 1):
                left, right = word[:cut], word[cut:]
                if not self._known(right):
                    continue
                heads = [left]
                if left[-1] in LINKS and len(left) > MIN_PART:
                    heads.append(left[:-1])
                for head in heads:
                    if self._known(head):
                        score = min(self.counts[head], self.counts[right])
                        if best is None or score > best[0]:
                            best = (score, head, right)
        parts: tuple[str, ...] = ()
        if best is not None:
            _, head, right = best
            parts = (head, *(self.split(right) or (right,)))
        self._cache[word] = parts
        return parts


def analyze(text: str, splitter: CompoundSplitter | None = None) -> list[str]:
    terms = []
    for word in TOKEN.findall(text.lower()):
        if word in STOP_WORDS or len(word) < 2:
            continue
        terms.append(stem(word))
        if splitter is not None:
            terms += [stem(part) for part in splitter.split(word) if part not in STOP_WORDS]
    return terms


class BM25Index:
    """BM25 weights in a sparse matrix, so that a query is one sum over a few columns."""

    def __init__(
        self,
        texts: list[str],
        splitter: CompoundSplitter | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.splitter = splitter
        self.vocabulary: dict[str, int] = {}
        rows, cols, counts = [], [], []
        lengths = np.zeros(len(texts), dtype=np.float64)
        for row, text in enumerate(texts):
            terms = Counter(analyze(text, splitter))
            lengths[row] = sum(terms.values())
            for term, count in terms.items():
                rows.append(row)
                cols.append(self.vocabulary.setdefault(term, len(self.vocabulary)))
                counts.append(count)
        tf = sparse.csr_matrix(
            (np.array(counts, dtype=np.float64), (rows, cols)),
            shape=(len(texts), len(self.vocabulary)),
        )
        df = np.bincount(tf.indices, minlength=len(self.vocabulary))
        n = len(texts)
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1e-9))
        row_of = np.repeat(np.arange(n), np.diff(tf.indptr))
        values = tf.data * (k1 + 1) / (tf.data + norm[row_of]) * idf[tf.indices]
        self.weights = sparse.csr_matrix((values, tf.indices, tf.indptr), shape=tf.shape).tocsc()
        self.size = n

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        columns = sorted(
            {self.vocabulary[t] for t in analyze(query, self.splitter) if t in self.vocabulary}
        )
        if not columns:
            return []
        scores = np.asarray(self.weights[:, columns].sum(axis=1)).ravel()
        k = min(k, self.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
