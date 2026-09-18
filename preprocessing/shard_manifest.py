"""
Shard Manifest for distributed training.

Provides a JSON-based manifest that catalogs all shards produced by a
preprocessing run, and deterministic shard-to-rank assignment for
PyTorch Distributed Data Parallel (DDP) compatibility.

The manifest is written alongside the shard files after a successful
preprocessing run and can be loaded by any training node to discover
which shards to read without filesystem scanning.

Manifest JSON format::

    {
        "version": 1,
        "created_at": "2026-09-12T17:30:00+00:00",
        "total_samples": 1000,
        "total_shards": 4,
        "total_bytes": 268435456,
        "shards": [
            {
                "path": "shard_0000.bin",
                "sample_count": 250,
                "size_bytes": 67108864,
                "checksum_crc32": "a1b2c3d4"
            }
        ]
    }
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import struct
import zlib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAGIC_START = b"VLMSHARD"
_HEADER_SIZE = 64


@dataclasses.dataclass
class ShardInfo:
    """Metadata for a single shard file."""

    path: str
    sample_count: int
    size_bytes: int
    checksum_crc32: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sample_count": self.sample_count,
            "size_bytes": self.size_bytes,
            "checksum_crc32": self.checksum_crc32,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardInfo:
        return cls(
            path=data["path"],
            sample_count=int(data["sample_count"]),
            size_bytes=int(data["size_bytes"]),
            checksum_crc32=data["checksum_crc32"],
        )


@dataclasses.dataclass
class ShardManifest:
    """Complete manifest for a set of shards."""

    version: int
    created_at: str
    total_samples: int
    total_shards: int
    total_bytes: int
    shards: list[ShardInfo]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "total_samples": self.total_samples,
            "total_shards": self.total_shards,
            "total_bytes": self.total_bytes,
            "shards": [s.to_dict() for s in self.shards],
        }

    def save(self, path: Path | str) -> None:
        """Write the manifest to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Saved shard manifest to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> ShardManifest:
        """Load a manifest from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        shards = [ShardInfo.from_dict(s) for s in data["shards"]]
        return cls(
            version=int(data["version"]),
            created_at=data["created_at"],
            total_samples=int(data["total_samples"]),
            total_shards=int(data["total_shards"]),
            total_bytes=int(data["total_bytes"]),
            shards=shards,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardManifest:
        """Construct from a plain dictionary."""
        shards = [ShardInfo.from_dict(s) for s in data.get("shards", [])]
        return cls(
            version=int(data.get("version", 1)),
            created_at=data.get("created_at", ""),
            total_samples=int(data.get("total_samples", 0)),
            total_shards=int(data.get("total_shards", 0)),
            total_bytes=int(data.get("total_bytes", 0)),
            shards=shards,
        )


def _parse_shard_header(path: Path) -> tuple[int, int]:
    """Read the sample count and format version from a shard header."""
    with open(path, "rb") as fh:
        header = fh.read(_HEADER_SIZE)

    if len(header) < _HEADER_SIZE:
        raise ValueError(
            f"Shard file {path} is too small ({len(header)} bytes, "
            f"expected at least {_HEADER_SIZE})"
        )

    magic = header[:8]
    if magic != _MAGIC_START:
        raise ValueError(
            f"Invalid shard magic in {path}: expected {_MAGIC_START!r}, "
            f"got {magic!r}"
        )

    version, sample_count = struct.unpack_from("<II", header, 8)
    return sample_count, version


def _compute_file_crc32(path: Path, exclude_footer: bool = True) -> str:
    """Compute CRC32 hex checksum of a shard file.

    If exclude_footer is True, the last 12 bytes (CRC32 + SHARDEND magic)
    are excluded, matching the shard writer's checksum calculation.
    """
    data = path.read_bytes()
    if exclude_footer and len(data) > 12:
        data = data[:-12]
    checksum = zlib.crc32(data) & 0xFFFFFFFF
    return f"{checksum:08x}"


def build_manifest(
    shard_dir: Path | str,
    *,
    glob_pattern: str = "shard_*.bin",
    compute_checksums: bool = True,
) -> ShardManifest:
    """Scan a directory and build a manifest from shard files.

    Parameters
    ----------
    shard_dir : Path | str
        Directory containing shard files.
    glob_pattern : str
        Glob pattern to match shard files.
    compute_checksums : bool
        If True, compute CRC32 checksums for each shard.

    Returns
    -------
    ShardManifest
        The constructed manifest.
    """
    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

    shard_files = sorted(shard_dir.glob(glob_pattern))
    if not shard_files:
        raise ValueError(
            f"No shard files matching '{glob_pattern}' found in {shard_dir}"
        )

    shards: list[ShardInfo] = []
    total_samples = 0
    total_bytes = 0

    for shard_path in shard_files:
        sample_count, _version = _parse_shard_header(shard_path)
        size_bytes = shard_path.stat().st_size

        checksum = ""
        if compute_checksums:
            checksum = _compute_file_crc32(shard_path)

        shards.append(ShardInfo(
            path=shard_path.name,
            sample_count=sample_count,
            size_bytes=size_bytes,
            checksum_crc32=checksum,
        ))

        total_samples += sample_count
        total_bytes += size_bytes

    manifest = ShardManifest(
        version=1,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        total_samples=total_samples,
        total_shards=len(shards),
        total_bytes=total_bytes,
        shards=shards,
    )

    logger.info(
        "Built manifest: %d shards, %d samples, %.2f MB",
        manifest.total_shards,
        manifest.total_samples,
        manifest.total_bytes / (1024 * 1024),
    )
    return manifest


def assign_shards(
    manifest: ShardManifest,
    rank: int,
    world_size: int,
) -> list[ShardInfo]:
    """Deterministically assign shards to a rank for distributed training.

    Uses round-robin assignment: shard i goes to rank (i % world_size).
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"rank must be in [0, {world_size}), got {rank}"
        )

    assigned = [
        shard
        for i, shard in enumerate(manifest.shards)
        if i % world_size == rank
    ]

    logger.info(
        "Rank %d/%d: assigned %d shards (%d samples)",
        rank,
        world_size,
        len(assigned),
        sum(s.sample_count for s in assigned),
    )
    return assigned
