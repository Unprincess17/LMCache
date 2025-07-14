import hashlib

import pytest
import torch
from utils import dumb_metadata, dumb_metadata_with_model_name, generate_tokens

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.token_database import (ChunkedTokenDatabase,
                                                 SegmentTokenDatabase,
                                                 LayerFirstTokenDataBase)
from lmcache.utils import LayerCacheEngineKey


@pytest.mark.parametrize('chunk_length', [16, 64, 256])
def test_chunked_token_database(chunk_length):
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_length,
                                          backend="cpu")
    metadata = dumb_metadata()

    test_length = 2500
    tokens = generate_tokens(test_length, "cpu")
    mask = torch.full([test_length], True, dtype=torch.bool, device="cpu")

    num_falses = [
        i * chunk_length for i in range(0, test_length // chunk_length)
    ]

    db = ChunkedTokenDatabase(cfg, metadata)

    # Process without mask
    original_results = list(db.process_tokens(tokens))
    for i in range(0, test_length, chunk_length):
        st, ed, key = original_results[i // chunk_length]
        assert st == i
        assert ed == min(i + chunk_length, test_length)

    for i in range(0, test_length // chunk_length):
        mask[:num_falses[i]] = False
        new_results = list(db.process_tokens(tokens, mask))
        assert len(new_results) == len(original_results) - i

        for j in range(len(new_results)):
            st, ed, key = new_results[j]
            assert st == original_results[j + i][0]
            assert ed == original_results[j + i][1]


@pytest.mark.parametrize('prefix_length', [0, 16, 64, 256])
@pytest.mark.parametrize('chunk_lengths', [[256, 512, 256], [1024, 512, 256]])
def test_segment_token_database(prefix_length, chunk_lengths):
    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=" # # ")
    metadata = dumb_metadata_with_model_name("facebook/opt-125m")

    db = SegmentTokenDatabase(cfg, metadata)
    sep_tokens = db.sep_tokens

    sys_length = 25
    query_length = 50
    sys_tokens = generate_tokens(sys_length, "cpu", fixed=True)
    query_tokens = generate_tokens(query_length, "cpu", fixed=True)

    token_chunks = []
    starts = [0]
    ends = [sys_length]
    sys_bytes = sys_tokens.cpu().to(torch.uint32).numpy().tobytes()
    sys_hash = hashlib.sha256(sys_bytes).hexdigest()
    hashes = [sys_hash]
    start = sys_length + len(sep_tokens)
    for idx, chunk_length in enumerate(chunk_lengths):
        token_chunk = generate_tokens(chunk_length, "cpu", fixed=True)

        token_bytes = token_chunk.cpu().to(torch.uint32).numpy().tobytes()
        token_hash = hashlib.sha256(token_bytes).hexdigest()
        hashes.append(token_hash)

        token_chunk = torch.cat([sep_tokens, token_chunk])
        token_chunks.append(token_chunk)
        starts.append(start)
        ends.append(start + chunk_length)
        start += chunk_length + len(sep_tokens)

    query_bytes = query_tokens.cpu().to(torch.uint32).numpy().tobytes()
    query_hash = hashlib.sha256(query_bytes).hexdigest()
    hashes.append(query_hash)

    tokens = torch.cat([sys_tokens, *token_chunks, sep_tokens, query_tokens])
    total_length = len(tokens)
    mask = torch.full([total_length], True, dtype=torch.bool, device="cpu")
    mask[:prefix_length] = False

    chunk_lists = [sys_tokens, *token_chunks, sep_tokens, query_tokens]
    skip_chunk_num = 0
    cum_length = 0
    for chunk in chunk_lists:
        if prefix_length > cum_length:
            skip_chunk_num += 1
        cum_length += len(chunk)

    starts = starts[skip_chunk_num:]
    ends = ends[skip_chunk_num:]
    hashes = hashes[skip_chunk_num:]

    original_results = list(db.process_tokens(tokens, mask))
    for i in range(len(original_results)):
        st, ed, key = original_results[i]
        assert st == starts[i]
        assert ed == ends[i]
        assert key.chunk_hash == hashes[i]
        #print(st, starts[i])
        #print(ed, ends[i])


@pytest.mark.parametrize('chunk_length', [16, 64, 256])
@pytest.mark.parametrize('num_layers', [2, 4, 8])
def test_layer_first_token_database(chunk_length, num_layers):
    """Test LayerFirstTokenDatabase basic functionality"""
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_length,
                                          backend="cpu")
    metadata = dumb_metadata()
    metadata.num_layers = num_layers

    test_length = 512
    tokens = generate_tokens(test_length, "cpu")

    db = LayerFirstTokenDataBase(cfg, metadata)

    # Process tokens
    results = list(db.process_tokens(tokens))

    # Calculate expected number of results
    num_chunks = (test_length + chunk_length - 1) // chunk_length
    expected_results = num_chunks * num_layers

    assert len(results) == expected_results

    # Verify that results are organized by chunk first, then by layer
    result_idx = 0
    for chunk_id in range(num_chunks):
        chunk_start = chunk_id * chunk_length
        chunk_end = min(chunk_start + chunk_length, test_length)

        for layer_id in range(num_layers):
            st, ed, key = results[result_idx]

            # Check start and end indices
            assert st == chunk_start
            assert ed == chunk_end

            # Check that key is LayerCacheEngineKey
            assert isinstance(key, LayerCacheEngineKey)
            assert key.layer_id == layer_id

            result_idx += 1


@pytest.mark.parametrize('chunk_length', [32, 128])
def test_layer_first_token_database_with_mask(chunk_length):
    """Test LayerFirstTokenDatabase with mask"""
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_length,
                                          backend="cpu")
    metadata = dumb_metadata()
    metadata.num_layers = 4

    test_length = 256
    tokens = generate_tokens(test_length, "cpu")

    # Create mask that masks out first chunk
    mask = torch.full([test_length], True, dtype=torch.bool, device="cpu")
    mask[:chunk_length] = False

    db = LayerFirstTokenDataBase(cfg, metadata)

    # Process tokens with mask
    results = list(db.process_tokens(tokens, mask))

    # Calculate expected number of results (excluding first chunk)
    num_chunks = (test_length + chunk_length - 1) // chunk_length
    expected_results = (num_chunks - 1) * metadata.num_layers

    assert len(results) == expected_results

    # Verify that first chunk is skipped
    for st, ed, key in results:
        assert st >= chunk_length  # All results should start after first chunk


def test_layer_first_token_database_hash_uniqueness():
    """Test that different layers have different hashes"""
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=64, backend="cpu")
    metadata = dumb_metadata()
    metadata.num_layers = 3

    test_length = 64  # Single chunk
    tokens = generate_tokens(test_length, "cpu")

    db = LayerFirstTokenDataBase(cfg, metadata)

    # Process tokens
    results = list(db.process_tokens(tokens))

    assert len(results) == 3  # One result per layer

    # Extract hashes from keys
    hashes = [key.chunk_hash for st, ed, key in results]

    # Verify all hashes are different
    assert len(set(hashes)) == 3, "All layer hashes should be unique"

    # Verify all results have same start/end but different layer_ids
    starts = [st for st, ed, key in results]
    ends = [ed for st, ed, key in results]
    layer_ids = [key.layer_id for st, ed, key in results]

    assert all(st == 0 for st in starts)
    assert all(ed == 64 for ed in ends)
    assert layer_ids == [0, 1, 2]


def test_layer_first_token_database_make_key_false():
    """Test LayerFirstTokenDatabase with make_key=False"""
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=32, backend="cpu")
    metadata = dumb_metadata()
    metadata.num_layers = 2

    test_length = 32  # Single chunk
    tokens = generate_tokens(test_length, "cpu")

    db = LayerFirstTokenDataBase(cfg, metadata)

    # Process tokens with make_key=False
    results = list(db.process_tokens(tokens, make_key=False))

    assert len(results) == 2  # One result per layer

    # Verify that third element is hash string, not key object
    for st, ed, hash_str in results:
        assert isinstance(hash_str, str)
        assert len(hash_str) == 64  # SHA256 hex digest length

    # Verify hashes are different for different layers
    hashes = [hash_str for st, ed, hash_str in results]
    assert len(set(hashes)) == 2, "Layer hashes should be unique"


def test_layer_first_token_database_list_input():
    """Test LayerFirstTokenDatabase with list input instead of tensor"""
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=16, backend="cpu")
    metadata = dumb_metadata()
    metadata.num_layers = 2

    # Use list instead of tensor
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    db = LayerFirstTokenDataBase(cfg, metadata)

    # Process tokens
    results = list(db.process_tokens(tokens))

    assert len(results) == 2  # One result per layer

    # Verify results
    for st, ed, key in results:
        assert st == 0
        assert ed == 16
        assert isinstance(key, LayerCacheEngineKey)
