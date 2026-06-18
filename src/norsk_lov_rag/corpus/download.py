"""Download Lovdata's open datasets through the public API.

Only the two public endpoints are used: the dataset list and the dataset download. Neither needs
a key. Files are kept as they arrive and identified by their SHA-256 hash, not by a date.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path

import httpx

from norsk_lov_rag.config import LovdataConfig

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _get(client: httpx.Client, url: str, max_retries: int) -> httpx.Response:
    """GET with exponential backoff and jitter on rate limits and server errors."""
    for attempt in range(max_retries + 1):
        try:
            response = client.get(url)
        except httpx.TransportError:
            if attempt == max_retries:
                raise
        else:
            if response.status_code not in RETRY_STATUS or attempt == max_retries:
                response.raise_for_status()
                return response
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                time.sleep(float(retry_after))
                continue
        time.sleep(min(60.0, 2.0**attempt) + random.uniform(0.0, 1.0))
    raise RuntimeError(f"no response from {url}")


def list_datasets(cfg: LovdataConfig, client: httpx.Client | None = None) -> list[dict]:
    """The datasets the API offers, as returned by the list endpoint."""
    owned = client is None
    client = client or httpx.Client(
        headers={"User-Agent": cfg.user_agent}, timeout=cfg.timeout_seconds
    )
    try:
        return _get(client, f"{cfg.base_url}/v1/publicData/list", cfg.max_retries).json()
    finally:
        if owned:
            client.close()


def download_datasets(cfg: LovdataConfig, raw_dir: Path, force: bool = False) -> dict[str, dict]:
    """Download every configured dataset that is not already on disk.

    Returns the file name, size in bytes and SHA-256 hash per dataset.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict] = {}
    with httpx.Client(
        headers={"User-Agent": cfg.user_agent}, timeout=cfg.timeout_seconds
    ) as client:
        offered = {row["filename"] for row in list_datasets(cfg, client)}
        for name in cfg.datasets:
            if name not in offered:
                raise ValueError(f"the API does not offer a dataset named {name}")
            target = raw_dir / name
            if force or not target.exists():
                response = _get(client, f"{cfg.base_url}/v1/publicData/get/{name}", cfg.max_retries)
                target.write_bytes(response.content)
                log.info("downloaded %s (%d bytes)", name, len(response.content))
            result[name] = {"bytes": target.stat().st_size, "sha256": sha256_of(target)}
    return result
