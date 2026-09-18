import pytest

from preprocessing.shard_manifest import ShardInfo, ShardManifest, assign_shards

def _mock_manifest(num_shards: int) -> ShardManifest:
    shards = [
        ShardInfo(
            path=f"shard_{i:04d}.bin",
            sample_count=100,
            size_bytes=1024,
            checksum_crc32="abcd",
        )
        for i in range(num_shards)
    ]
    return ShardManifest(
        version=1,
        created_at="2026-09-18",
        total_samples=100 * num_shards,
        total_shards=num_shards,
        total_bytes=1024 * num_shards,
        shards=shards,
    )

def test_determinism():
    """1. Determinism: Running twice with same inputs yields exact same list."""
    manifest = _mock_manifest(10)
    res1 = assign_shards(manifest, rank=1, world_size=4)
    res2 = assign_shards(manifest, rank=1, world_size=4)
    assert [s.path for s in res1] == [s.path for s in res2]

def test_total_coverage():
    """2. Total coverage: Union of shards across all ranks equals total set."""
    manifest = _mock_manifest(10)
    world_size = 4
    all_assigned = []
    for r in range(world_size):
        all_assigned.extend(assign_shards(manifest, rank=r, world_size=world_size))

    assigned_paths = {s.path for s in all_assigned}
    expected_paths = {s.path for s in manifest.shards}
    assert assigned_paths == expected_paths
    assert len(all_assigned) == len(manifest.shards) # no duplicates globally

def test_no_overlap():
    """3. No overlap: Intersection between any two ranks is empty."""
    manifest = _mock_manifest(10)
    res0 = assign_shards(manifest, rank=0, world_size=3)
    res1 = assign_shards(manifest, rank=1, world_size=3)
    paths0 = {s.path for s in res0}
    paths1 = {s.path for s in res1}
    assert paths0.isdisjoint(paths1)

def test_single_rank_edge_case():
    """4. Single rank edge case: W=1 gets all shards."""
    manifest = _mock_manifest(10)
    res = assign_shards(manifest, rank=0, world_size=1)
    assert len(res) == 10
    assert [s.path for s in res] == [s.path for s in manifest.shards]

def test_ranks_greater_than_files():
    """5. Ranks > Files edge case: W > K."""
    manifest = _mock_manifest(2)
    world_size = 4
    r0 = assign_shards(manifest, rank=0, world_size=world_size) # should get 1
    r1 = assign_shards(manifest, rank=1, world_size=world_size) # should get 1
    r2 = assign_shards(manifest, rank=2, world_size=world_size) # should get 0

    assert len(r0) == 1
    assert len(r1) == 1
    assert len(r2) == 0

def test_perfectly_balanced():
    """6. Perfectly balanced distribution: K is multiple of W."""
    manifest = _mock_manifest(12)
    world_size = 4
    for r in range(world_size):
        res = assign_shards(manifest, rank=r, world_size=world_size)
        assert len(res) == 3

def test_imbalanced_distribution():
    """7. Imbalanced distribution (Delta <= 1)."""
    manifest = _mock_manifest(11)
    world_size = 4
    counts = []
    for r in range(world_size):
        res = assign_shards(manifest, rank=r, world_size=world_size)
        counts.append(len(res))

    max_count = max(counts)
    min_count = min(counts)
    assert max_count - min_count <= 1

def test_explicit_rejection_negative_rank():
    """8. Explicit rejection 1: rank < 0."""
    manifest = _mock_manifest(5)
    with pytest.raises(ValueError):
        assign_shards(manifest, rank=-1, world_size=4)

def test_explicit_rejection_rank_out_of_bounds():
    """9. Explicit rejection 2: rank >= total_ranks."""
    manifest = _mock_manifest(5)
    with pytest.raises(ValueError):
        assign_shards(manifest, rank=4, world_size=4)

def test_explicit_rejection_invalid_world_size():
    """10. Explicit rejection 3: total_ranks <= 0."""
    manifest = _mock_manifest(5)
    with pytest.raises(ValueError):
        assign_shards(manifest, rank=0, world_size=0)
