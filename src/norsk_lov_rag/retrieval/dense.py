"""Dense retrieval with sentence-transformers models on the CPU.

Chunk embeddings are written in shards, so a long run can stop and go on later without losing
work. Each shard records how long it took, and the index records its total embedding time and
its size on disk.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from norsk_lov_rag.config import EncoderConfig, RerankerConfig

SHARD_ROWS = 2048


def _hub(model_dir: Path) -> None:
    """Keep downloaded models inside the project's ignored model folder."""
    os.environ.setdefault("HF_HUB_CACHE", str(model_dir.resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def load_encoder(spec: EncoderConfig, model_dir: Path):
    _hub(model_dir)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec.hf_id, revision=spec.revision, device="cpu")
    model.max_seq_length = spec.max_seq_length
    return model


def encode(model, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    vectors = model.encode(
        [prefix + text for text in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


def index_dir(data_dir: Path, scope: str, strategy: str) -> Path:
    return data_dir / "index" / f"{scope}-{strategy}"


def build_dense_index(
    spec: EncoderConfig, name: str, texts: list[str], folder: Path, model_dir: Path, log=print
) -> dict:
    """Embed every chunk, shard by shard, and join the shards into one matrix."""
    final = folder / f"{name}.npy"
    meta_path = folder / f"{name}.json"
    if final.exists() and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    shards = folder / f"{name}-shards"
    shards.mkdir(parents=True, exist_ok=True)
    model = None
    count = (len(texts) + SHARD_ROWS - 1) // SHARD_ROWS
    for number in range(count):
        path = shards / f"{number:05d}.npy"
        if path.exists():
            continue
        if model is None:
            model = load_encoder(spec, model_dir)
        start = time.perf_counter()
        part = texts[number * SHARD_ROWS : (number + 1) * SHARD_ROWS]
        vectors = encode(model, part, spec.passage_prefix, spec.batch_size)
        seconds = time.perf_counter() - start
        np.save(path, vectors)
        (shards / f"{number:05d}.json").write_text(json.dumps({"seconds": seconds}))
        log(f"{name}: shard {number + 1} of {count}, {len(part) / seconds:.1f} chunks per second")
    matrix = np.concatenate([np.load(shards / f"{n:05d}.npy") for n in range(count)])
    seconds = sum(
        json.loads((shards / f"{n:05d}.json").read_text())["seconds"] for n in range(count)
    )
    np.save(final, matrix)
    meta = {
        "encoder": name,
        "hf_id": spec.hf_id,
        "revision": spec.revision,
        "rows": int(matrix.shape[0]),
        "dimensions": int(matrix.shape[1]),
        "embedding_seconds": round(seconds, 1),
        "index_bytes": final.stat().st_size,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for path in shards.iterdir():
        path.unlink()
    shards.rmdir()
    return meta


class DenseIndex:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    @classmethod
    def load(cls, folder: Path, name: str) -> DenseIndex:
        return cls(np.load(folder / f"{name}.npy", mmap_mode="r"))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores = np.asarray(self.matrix @ query).ravel()
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top]


def load_reranker(spec: RerankerConfig, model_dir: Path):
    _hub(model_dir)
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        spec.hf_id, revision=spec.revision, max_length=spec.max_length, device="cpu"
    )


def rerank(model, query: str, texts: list[str], batch_size: int) -> list[float]:
    scores = model.predict(
        [(query, text) for text in texts], batch_size=batch_size, show_progress_bar=False
    )
    return [float(score) for score in scores]
