import json
import struct
from pathlib import Path

import pytest

from preprocessing.shard_manifest import ShardManifest, build_manifest, _MAGIC_START

def _create_dummy_shard(path: Path, sample_count: int, version: int = 2) -> None:
    # 64 byte header
    header = struct.pack("<8sII", _MAGIC_START, version, sample_count)
    # Pad to 64 bytes
    header += b"\x00" * (64 - len(header))

    # Write some dummy payload so file has size and crc can be computed
    payload = b"dummy_data_for_crc32"

    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)

def test_manifest_io_roundtrip(tmp_path: Path):
    """1. I/O Round-trip: Creates a manifest, writes it to disk, reads it back, asserts equality."""
    from preprocessing.shard_manifest import ShardInfo

    manifest = ShardManifest(
        version=1,
        created_at="2026-09-18",
        total_samples=100,
        total_shards=1,
        total_bytes=2048,
        shards=[
            ShardInfo(path="shard_0000.bin", sample_count=100, size_bytes=2048, checksum_crc32="abcd")
        ]
    )

    out_file = tmp_path / "manifest.json"
    manifest.save(out_file)

    loaded = ShardManifest.load(out_file)

    assert manifest.version == loaded.version
    assert manifest.created_at == loaded.created_at
    assert manifest.total_samples == loaded.total_samples
    assert manifest.total_shards == loaded.total_shards
    assert manifest.total_bytes == loaded.total_bytes
    assert len(manifest.shards) == len(loaded.shards)
    assert manifest.shards[0].path == loaded.shards[0].path

def test_exact_aggregated_total_samples(tmp_path: Path):
    """2. Exact aggregated total samples: Sums samples across shards and matches global total."""
    d = tmp_path / "shards"
    d.mkdir()
    _create_dummy_shard(d / "shard_0000.bin", 50)
    _create_dummy_shard(d / "shard_0001.bin", 150)

    manifest = build_manifest(d)

    assert manifest.total_shards == 2
    assert manifest.total_samples == 200
    assert sum(s.sample_count for s in manifest.shards) == 200

def test_json_schema_validation(tmp_path: Path):
    """3. JSON Schema validation: Asserts the produced JSON strictly follows the expected structure."""
    d = tmp_path / "shards"
    d.mkdir()
    _create_dummy_shard(d / "shard_0000.bin", 10)

    manifest = build_manifest(d)
    out_file = tmp_path / "manifest.json"
    manifest.save(out_file)

    with open(out_file, "r") as f:
        data = json.load(f)

    assert "version" in data and isinstance(data["version"], int)
    assert "created_at" in data and isinstance(data["created_at"], str)
    assert "total_samples" in data and isinstance(data["total_samples"], int)
    assert "total_shards" in data and isinstance(data["total_shards"], int)
    assert "total_bytes" in data and isinstance(data["total_bytes"], int)
    assert "shards" in data and isinstance(data["shards"], list)

    shard_info = data["shards"][0]
    assert "path" in shard_info and isinstance(shard_info["path"], str)
    assert "sample_count" in shard_info and isinstance(shard_info["sample_count"], int)
    assert "size_bytes" in shard_info and isinstance(shard_info["size_bytes"], int)
    assert "checksum_crc32" in shard_info and isinstance(shard_info["checksum_crc32"], str)

def test_explicit_rejection_empty_directory(tmp_path: Path):
    """4. Explicit rejection 1: Raises an expected error when pointing to an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No shard files matching"):
        build_manifest(empty_dir)

def test_explicit_rejection_non_existent_directory(tmp_path: Path):
    """5. Explicit rejection 2: Raises an expected error when pointing to a non-existent directory."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="Shard directory not found"):
        build_manifest(non_existent)
