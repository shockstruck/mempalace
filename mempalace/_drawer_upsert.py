"""Adaptive batching for ChromaDB drawer upserts.

Both ``convo_miner`` and ``miner`` batch chunks into ``collection.upsert``
calls so the embedding model sees many chunks per forward pass and the
HTTP client makes proportionally fewer requests. A record-count cap
(``DRAWER_UPSERT_BATCH_SIZE = 1000``) is sufficient when chunk content
is uniformly small (project-mode at ``CHUNK_SIZE = 800`` chars), but
the convo extractor's ``general`` mode can produce a single chunk of
50–100 KB when a chat exchange contains a multi-thousand-line code
paste. With 1000 such chunks per batch that's a 50–100 MB JSON body,
which Chroma rejects with ``Payload too large`` regardless of how
many records you sent.

This module yields slices bounded by *both* the record count AND a
byte estimate, so byte-heavy batches flush early without affecting the
small-chunk fast path. Both limits are env-overridable:

* ``MEMPALACE_DRAWER_UPSERT_BATCH_SIZE`` — record cap (default 1000)
* ``MEMPALACE_DRAWER_UPSERT_MAX_BYTES``  — JSON body cap (default 8 MiB)
"""

from __future__ import annotations

import os
from typing import Iterator, Sequence

# Per-drawer JSON overhead (excluding the document content itself):
#
# * ``embeddings``: 384 float32 values; chromadb sends them as JSON ASCII
#   floats. ~15 chars/value × 384 = ~5,760 bytes plus list delimiters.
# * ``ids``:        the ``drawer_<wing>_<room>_<24-hex>`` form is ~80 bytes.
# * ``metadatas``:  ~10 keys × (key + short string value + JSON quoting +
#                   commas) ≈ 400–600 bytes per record.
# * Outer wrapping: per-record commas, brackets, and field separators in
#                   the parallel-list shape chromadb's HTTP API uses.
#
# 6500 is the conservative upper bound — flushing slightly early is
# cheaper than blowing the server's body limit and aborting the file.
_PER_DRAWER_OVERHEAD_BYTES = 6500

DEFAULT_MAX_COUNT = 1000
DEFAULT_MAX_BYTES = 8 * 1024 * 1024  # 8 MiB


def _estimated_drawer_bytes(content: str) -> int:
    """Approximate JSON bytes a single chunk occupies in the upsert payload.

    The 5%-bump (``len // 20``) accounts for JSON escape sequences in
    ASCII-heavy content; not exact, but conservative.
    """
    return _PER_DRAWER_OVERHEAD_BYTES + len(content) + (len(content) // 20)


def _resolve_int_env(name: str, default: int) -> int:
    """Read ``name`` from the environment as a positive int, else ``default``.

    Invalid / non-positive values fall back to ``default`` silently — the
    miner's job is to make progress, not to validate operator typos.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        return default
    return n if n > 0 else default


def get_max_count() -> int:
    """Per-batch record cap, overridable via ``MEMPALACE_DRAWER_UPSERT_BATCH_SIZE``."""
    return _resolve_int_env("MEMPALACE_DRAWER_UPSERT_BATCH_SIZE", DEFAULT_MAX_COUNT)


def get_max_bytes() -> int:
    """Per-batch byte budget, overridable via ``MEMPALACE_DRAWER_UPSERT_MAX_BYTES``."""
    return _resolve_int_env("MEMPALACE_DRAWER_UPSERT_MAX_BYTES", DEFAULT_MAX_BYTES)


def iter_batches(
    chunks: Sequence,
    *,
    content_key: str = "content",
    max_count: int | None = None,
    max_bytes: int | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` index pairs partitioning ``chunks`` for upsert.

    Each yielded slice ``chunks[start:end]`` satisfies both
    ``end - start <= max_count`` and (estimated) ``sum(bytes) <= max_bytes``,
    flushing whichever bound is hit first. The trailing partial batch is
    always emitted.

    A single chunk whose own estimated size exceeds ``max_bytes`` is
    yielded as a 1-record batch and left to Chroma to accept or reject —
    silently dropping data is the worse failure mode.

    ``content_key`` lets callers disambiguate between the convo miner's
    chunk shape (``{"content": ..., "chunk_index": ...}``) and the
    project miner's identical shape; default works for both today.
    """
    if max_count is None:
        max_count = get_max_count()
    if max_bytes is None:
        max_bytes = get_max_bytes()

    n = len(chunks)
    start = 0
    running_bytes = 0
    cursor = 0

    while cursor < n:
        content = chunks[cursor].get(content_key, "") if hasattr(chunks[cursor], "get") else ""
        chunk_bytes = _estimated_drawer_bytes(content)

        # Flush before adding when this chunk would push us over the byte
        # budget AND we've already accumulated at least one chunk in the
        # current batch — otherwise an oversized lone chunk would loop.
        if running_bytes + chunk_bytes > max_bytes and cursor > start:
            yield (start, cursor)
            start = cursor
            running_bytes = 0
            continue

        running_bytes += chunk_bytes
        cursor += 1

        if cursor - start >= max_count:
            yield (start, cursor)
            start = cursor
            running_bytes = 0

    if start < n:
        yield (start, n)
