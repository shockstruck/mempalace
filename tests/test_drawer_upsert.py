"""Tests for ``mempalace._drawer_upsert.iter_batches``.

Covers both bounds (count cap, byte cap), env-var overrides, and the
edge case of a lone oversized chunk being yielded as its own 1-record
batch rather than getting silently skipped.
"""

from __future__ import annotations

import pytest

from mempalace._drawer_upsert import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_COUNT,
    _estimated_drawer_bytes,
    get_max_bytes,
    get_max_count,
    iter_batches,
)


def _chunk(content: str, idx: int = 0) -> dict:
    """Minimal chunk shape — both miners pass dicts with a ``content`` key."""
    return {"content": content, "chunk_index": idx}


# ── Count-bound (small chunks: legacy fast path) ─────────────────────────


def test_count_bound_emits_full_batches_then_partial():
    chunks = [_chunk("x" * 10, i) for i in range(2500)]
    batches = list(iter_batches(chunks, max_count=1000, max_bytes=10**9))
    assert batches == [(0, 1000), (1000, 2000), (2000, 2500)]


def test_count_bound_exact_multiple_no_trailing_empty_batch():
    chunks = [_chunk("x", i) for i in range(2000)]
    batches = list(iter_batches(chunks, max_count=1000, max_bytes=10**9))
    assert batches == [(0, 1000), (1000, 2000)]


def test_empty_input_yields_nothing():
    assert list(iter_batches([], max_count=1000, max_bytes=10**9)) == []


def test_single_chunk_yields_single_batch():
    chunks = [_chunk("hello", 0)]
    assert list(iter_batches(chunks, max_count=1000, max_bytes=10**9)) == [(0, 1)]


# ── Byte-bound (the convo --extract general failure mode) ────────────────


def test_byte_bound_flushes_before_count_when_chunks_are_large():
    """80 KB chunks * 100 records ≈ 8 MB → byte cap hits before count cap."""
    big = "x" * 80_000  # 80 KB content
    chunks = [_chunk(big, i) for i in range(100)]
    # Cap at 1 MiB to force byte-bound behavior
    batches = list(iter_batches(chunks, max_count=1000, max_bytes=1 * 1024 * 1024))
    # Each chunk's est bytes ≈ 80_000 + 6500 + 4000 = ~90.5 KB
    # → about 11–12 chunks per 1 MiB batch
    assert len(batches) > 1, "expected multiple batches under byte cap"
    for start, end in batches:
        # Every batch except the last must be < max_count
        assert end - start <= 1000
        # And every batch must respect the byte cap (or be a 1-record
        # oversized chunk, which isn't the case here)
        approx_bytes = sum(_estimated_drawer_bytes(chunks[i]["content"]) for i in range(start, end))
        if end - start > 1:
            assert approx_bytes <= 1 * 1024 * 1024, f"batch {start}-{end} = {approx_bytes} bytes"


def test_byte_bound_lone_oversized_chunk_yielded_as_singleton():
    """A chunk bigger than the byte cap on its own must still be yielded —
    silently skipping data is a worse failure than letting Chroma reject."""
    huge = "x" * (5 * 1024 * 1024)  # 5 MB, larger than 1 MiB cap
    small = "y" * 100
    chunks = [_chunk(small, 0), _chunk(huge, 1), _chunk(small, 2)]
    batches = list(iter_batches(chunks, max_count=1000, max_bytes=1 * 1024 * 1024))
    # Expected: [(0,1)=small, (1,2)=huge alone, (2,3)=small]
    assert batches == [(0, 1), (1, 2), (2, 3)]


def test_byte_bound_mixed_sizes_flush_before_oversized():
    """Small chunks accumulate; arrival of a chunk that would push the batch
    over the byte cap triggers an early flush before that chunk."""
    small = "y" * 100
    big = "x" * 80_000  # 80 KB
    # 5 small chunks then one big one
    chunks = [_chunk(small, i) for i in range(5)] + [_chunk(big, 5)]
    # Cap chosen so 5 smalls fit but 5 smalls + 1 big does not.
    # 5 smalls ≈ 5 × (6500 + 100) = ~33 KB
    # Adding big = 33 KB + ~84 KB = ~117 KB
    # Set cap at 50 KB — smalls fit (33 KB), big alone exceeds (84 KB).
    batches = list(iter_batches(chunks, max_count=1000, max_bytes=50 * 1024))
    # Expected: smalls flush as (0,5), then big alone as (5,6)
    assert batches == [(0, 5), (5, 6)]


# ── Env var overrides ────────────────────────────────────────────────────


def test_env_overrides_max_count(monkeypatch):
    monkeypatch.setenv("MEMPALACE_DRAWER_UPSERT_BATCH_SIZE", "50")
    assert get_max_count() == 50
    chunks = [_chunk("x", i) for i in range(120)]
    batches = list(iter_batches(chunks, max_bytes=10**9))
    assert batches == [(0, 50), (50, 100), (100, 120)]


def test_env_overrides_max_bytes(monkeypatch):
    monkeypatch.setenv("MEMPALACE_DRAWER_UPSERT_MAX_BYTES", str(20 * 1024))
    assert get_max_bytes() == 20 * 1024
    big = "x" * 80_000
    chunks = [_chunk(big, i) for i in range(3)]
    batches = list(iter_batches(chunks, max_count=1000))
    # Each chunk far exceeds 20 KB → each yielded as singleton
    assert batches == [(0, 1), (1, 2), (2, 3)]


def test_env_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MEMPALACE_DRAWER_UPSERT_BATCH_SIZE", "not-a-number")
    assert get_max_count() == DEFAULT_MAX_COUNT
    monkeypatch.setenv("MEMPALACE_DRAWER_UPSERT_BATCH_SIZE", "0")
    assert get_max_count() == DEFAULT_MAX_COUNT
    monkeypatch.setenv("MEMPALACE_DRAWER_UPSERT_BATCH_SIZE", "-5")
    assert get_max_count() == DEFAULT_MAX_COUNT


def test_env_unset_uses_defaults(monkeypatch):
    monkeypatch.delenv("MEMPALACE_DRAWER_UPSERT_BATCH_SIZE", raising=False)
    monkeypatch.delenv("MEMPALACE_DRAWER_UPSERT_MAX_BYTES", raising=False)
    assert get_max_count() == DEFAULT_MAX_COUNT
    assert get_max_bytes() == DEFAULT_MAX_BYTES


# ── Sanity on the byte estimator ─────────────────────────────────────────


def test_estimated_drawer_bytes_grows_with_content():
    small = _estimated_drawer_bytes("")
    big = _estimated_drawer_bytes("x" * 10_000)
    assert big > small
    # Overhead floor: even empty content costs the per-drawer overhead
    assert small >= 6000


@pytest.mark.parametrize("content_size", [0, 100, 1_000, 50_000])
def test_estimated_drawer_bytes_monotonic(content_size):
    base = _estimated_drawer_bytes("x" * content_size)
    bigger = _estimated_drawer_bytes("x" * (content_size + 1))
    assert bigger >= base
