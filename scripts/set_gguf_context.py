"""Copy a GGUF file and set its context length, without touching any weights.

The GGUF file of NorMistral-7b-warm-instruct says 2048 tokens of context, which is the length
of the base model. The model card says the instruction model was tuned on 4096 tokens, and the
brief plans for that window. Ollama never uses more than the file says, so this script writes a
copy where only the number ``llama.context_length`` is changed.

Usage: python scripts/set_gguf_context.py SOURCE.gguf TARGET.gguf 4096
"""

from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

KEY = b"llama.context_length"
UINT32 = 4


def main(source: Path, target: Path, length: int) -> None:
    with source.open("rb") as handle:
        head = handle.read(16 * 1024 * 1024)
    if head[:4] != b"GGUF":
        raise SystemExit(f"{source} is not a GGUF file")
    # Each key is stored as a length-prefixed string, followed by the value type and the value.
    marker = struct.pack("<Q", len(KEY)) + KEY
    place = head.find(marker)
    if place < 0 or head.find(marker, place + 1) >= 0:
        raise SystemExit("the context length key was not found exactly once")
    offset = place + len(marker)
    kind, old = struct.unpack_from("<II", head, offset)
    if kind != UINT32:
        raise SystemExit(f"unexpected value type {kind}")
    shutil.copyfile(source, target)
    with target.open("r+b") as handle:
        handle.seek(offset + 4)
        handle.write(struct.pack("<I", length))
    print(f"{KEY.decode()}: {old} -> {length} in {target}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
