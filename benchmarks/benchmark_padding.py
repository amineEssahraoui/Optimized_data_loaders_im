#!/usr/bin/env python3
"""
benchmark_padding.py — Static vs. Dynamic Padding Benchmark (Real Data)
=========================================================================

Compares two multimodal padding strategies on REAL HuggingFace VQA images:

  Static Padding  (Format v1):
    Images  → resize_and_pad to fixed 384×384, float32 storage
    Text    → all sequences padded to global max_seq_length (512 tokens)

  Dynamic Padding (Format v2):
    Images  → aspect-ratio-preserving resize, uint8 storage (≤384px max dim)
    Text    → per-batch max(actual_seq_len) using attention masks

All measurements are on REAL images with realistic dimension distributions.

Output (each metric = its own standalone PNG in benchmarks/results/):

  static_vs_dynamic_storage.png
        On-disk total storage footprint (MB) — image + text breakdown
        Shows absolute numbers and percentage reduction.

  static_vs_dynamic_batch_memory.png
        Per-batch GPU memory consumption (MB, float32 in-memory) —
        image tensor + token arrays, averaged across simulated batches.

  static_vs_dynamic_vit_flops.png
        ViT FLOPs proxy: average effective H×W per batch
        (ViT attention is O(patches²) ∝ H×W for feedforward path).

  static_vs_dynamic_llm_flops.png
        LLM FLOPs proxy: average effective sequence length per batch
        (transformer attention is O(seq_len²)).

  static_vs_dynamic_throughput.png
        Shard read throughput (samples/second) — ShardReader simulating
        how fast the C++ loader serves data.

  static_vs_dynamic_latency.png
        Per-batch read latency distribution (box plot with P50/P99 markers).

Usage::

    # Quick run with 500 real samples
    python benchmarks/benchmark_padding.py --num-samples 500

    # Full production run
    python benchmarks/benchmark_padding.py \\
        --num-samples 2000 \\
        --batch-size 16 \\
        --num-batches 80 \\
        --output-dir benchmarks/results

    # Use pre-built shards (skip shard generation)
    python benchmarks/benchmark_padding.py \\
        --static-shards output/shards_v1/ \\
        --dynamic-shards output/shards_v2/
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap imports
# ---------------------------------------------------------------------------
_BENCH_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCH_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.bench_utils import (  # noqa: E402
    BRIGHT_PALETTE,
    COLOR_DYNAMIC,
    COLOR_IMAGE,
    COLOR_SAVINGS,
    COLOR_SPEEDUP,
    COLOR_STATIC,
    COLOR_TEXT,
    add_bar_labels,
    apply_bright_style,
    load_real_dataset,
    save_fig,
)
from preprocessing.config import (  # noqa: E402
    ImageConfig,
    PipelineConfig,
    ShardConfig,
    TokenizerConfig,
)
from preprocessing.image_processor import normalize_image, resize_preserve_aspect  # noqa: E402
from preprocessing.schema import VQASample  # noqa: E402
from preprocessing.shard_reader import ShardReader  # noqa: E402
from preprocessing.shard_writer import ShardWriter  # noqa: E402
from preprocessing.tokenizer import TextTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = _BENCH_DIR / "results"

STATIC_IMG_H    = 384
STATIC_IMG_W    = 384
DYNAMIC_MAX_DIM = 384
MAX_SEQ_LENGTH  = 512
NUM_CHANNELS    = 3
NUM_TOKEN_ARRAYS = 4  # q_ids, q_mask, a_ids, a_mask
ELEM_SIZE_F32   = 4   # bytes per float32
ELEM_SIZE_INT32 = 4   # bytes per int32
ELEM_SIZE_UINT8 = 1   # bytes per uint8

_TOKENIZER_ID = "bert-base-uncased"


# ---------------------------------------------------------------------------
# Analysis dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StorageMetrics:
    """On-disk storage footprint analysis."""
    strategy: str
    num_samples: int = 0
    total_image_bytes: int = 0
    total_text_bytes: int = 0
    total_overhead_bytes: int = 0
    image_bytes_per_sample: list[int] = field(default_factory=list)
    text_bytes_per_sample: list[int] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return (self.total_image_bytes
                + self.total_text_bytes
                + self.total_overhead_bytes)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)

    @property
    def image_mb(self) -> float:
        return self.total_image_bytes / (1024 ** 2)

    @property
    def text_mb(self) -> float:
        return self.total_text_bytes / (1024 ** 2)


@dataclass
class BatchMetrics:
    """Per-batch memory and timing analysis."""
    strategy: str
    batch_size: int = 0
    num_batches: int = 0
    image_memory_bytes: list[int] = field(default_factory=list)
    text_memory_bytes: list[int] = field(default_factory=list)
    batch_latencies_ms: list[float] = field(default_factory=list)
    batch_img_heights: list[int] = field(default_factory=list)
    batch_img_widths: list[int] = field(default_factory=list)
    batch_max_seq_lens: list[int] = field(default_factory=list)
    total_time_s: float = 0.0
    total_samples: int = 0

    @property
    def avg_image_mem_mb(self) -> float:
        return np.mean(self.image_memory_bytes) / (1024 ** 2) if self.image_memory_bytes else 0.0

    @property
    def avg_text_mem_mb(self) -> float:
        return np.mean(self.text_memory_bytes) / (1024 ** 2) if self.text_memory_bytes else 0.0

    @property
    def avg_total_mem_mb(self) -> float:
        return self.avg_image_mem_mb + self.avg_text_mem_mb

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.batch_latencies_ms)) if self.batch_latencies_ms else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return float(np.percentile(self.batch_latencies_ms, 50)) if self.batch_latencies_ms else 0.0

    @property
    def p99_latency_ms(self) -> float:
        return float(np.percentile(self.batch_latencies_ms, 99)) if self.batch_latencies_ms else 0.0

    @property
    def throughput_sps(self) -> float:
        return self.total_samples / self.total_time_s if self.total_time_s > 0 else 0.0

    @property
    def avg_vit_hw(self) -> float:
        """Average H×W product per batch — ViT FLOPs proxy."""
        if not self.batch_img_heights:
            return 0.0
        return float(np.mean([
            h * w for h, w in zip(self.batch_img_heights, self.batch_img_widths)
        ]))

    @property
    def avg_seq_len(self) -> float:
        """Average max sequence length per batch — LLM FLOPs proxy."""
        return float(np.mean(self.batch_max_seq_lens)) if self.batch_max_seq_lens else 0.0


# ---------------------------------------------------------------------------
# Shard writing
# ---------------------------------------------------------------------------

def write_static_shards(
    samples: list[VQASample],
    output_dir: Path,
    tokenizer: TextTokenizer,
) -> None:
    """Write Format v1 (static padding, float32, 384×384) shards."""
    output_dir.mkdir(parents=True, exist_ok=True)

    img_cfg = ImageConfig(
        target_size=(STATIC_IMG_H, STATIC_IMG_W),
        resize_strategy="resize_and_pad",
        storage_dtype="float32",
        dynamic_padding=False,
        color_space="RGB",
    )
    shard_cfg = ShardConfig(
        output_dir=str(output_dir),
        format_version=1,
        shard_size_mb=512,
        max_samples_per_shard=500,
    )
    tok_cfg = TokenizerConfig(
        model_name_or_path=_TOKENIZER_ID,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        trust_remote_code=False,
        add_special_tokens=True,
    )

    writer = ShardWriter(shard_cfg, img_cfg, tok_cfg)
    shard_path = output_dir / "shard_0000.bin"
    writer.open(shard_path)

    shard_index = 0
    for i, sample in enumerate(samples):
        try:
            image_tensor = normalize_image(sample.image_bytes, img_cfg)
            q_tok = tokenizer.tokenize(sample.question)
            a_tok = tokenizer.tokenize(sample.answer)

            writer.add_sample(
                image_tensor=image_tensor,
                question_ids=q_tok.input_ids,
                question_mask=q_tok.attention_mask,
                answer_ids=a_tok.input_ids,
                answer_mask=a_tok.attention_mask,
                metadata={"sample_id": sample.sample_id, "strategy": "static"},
            )

            if writer.should_rotate():
                writer.close()
                shard_index += 1
                shard_path = output_dir / f"shard_{shard_index:04d}.bin"
                writer = ShardWriter(shard_cfg, img_cfg, tok_cfg)
                writer.open(shard_path)

        except Exception as exc:
            if i < 5:
                print(f"    [static] skipping sample {i}: {exc}")
            continue

    writer.close()
    shard_count = shard_index + 1
    total_mb = sum(
        Path(p).stat().st_size for p in output_dir.glob("shard_*.bin")
    ) / (1024 ** 2)
    print(f"  ✓ Static shards: {shard_count} files, {total_mb:.1f} MB total")


def write_dynamic_shards(
    samples: list[VQASample],
    output_dir: Path,
    tokenizer: TextTokenizer,
) -> None:
    """Write Format v2 (dynamic padding, uint8, aspect-ratio) shards."""
    output_dir.mkdir(parents=True, exist_ok=True)

    img_cfg = ImageConfig(
        max_image_dim=DYNAMIC_MAX_DIM,
        storage_dtype="uint8",
        dynamic_padding=True,
        color_space="RGB",
        interpolation="bilinear",
    )
    shard_cfg = ShardConfig(
        output_dir=str(output_dir),
        format_version=2,
        shard_size_mb=512,
        max_samples_per_shard=500,
    )
    tok_cfg = TokenizerConfig(
        model_name_or_path=_TOKENIZER_ID,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        trust_remote_code=False,
        add_special_tokens=True,
    )

    writer = ShardWriter(shard_cfg, img_cfg, tok_cfg)
    shard_path = output_dir / "shard_0000.bin"
    writer.open(shard_path)

    shard_index = 0
    for i, sample in enumerate(samples):
        try:
            img_arr, orig_h, orig_w, actual_h, actual_w = resize_preserve_aspect(
                sample.image_bytes, img_cfg
            )
            q_tok = tokenizer.tokenize(sample.question)
            a_tok = tokenizer.tokenize(sample.answer)

            writer.add_sample(
                image_tensor=img_arr,
                question_ids=q_tok.input_ids,
                question_mask=q_tok.attention_mask,
                answer_ids=a_tok.input_ids,
                answer_mask=a_tok.attention_mask,
                metadata={"sample_id": sample.sample_id, "strategy": "dynamic"},
                orig_height=orig_h,
                orig_width=orig_w,
            )

            if writer.should_rotate():
                writer.close()
                shard_index += 1
                shard_path = output_dir / f"shard_{shard_index:04d}.bin"
                writer = ShardWriter(shard_cfg, img_cfg, tok_cfg)
                writer.open(shard_path)

        except Exception as exc:
            if i < 5:
                print(f"    [dynamic] skipping sample {i}: {exc}")
            continue

    writer.close()
    shard_count = shard_index + 1
    total_mb = sum(
        Path(p).stat().st_size for p in output_dir.glob("shard_*.bin")
    ) / (1024 ** 2)
    print(f"  ✓ Dynamic shards: {shard_count} files, {total_mb:.1f} MB total")


# ---------------------------------------------------------------------------
# Storage analysis
# ---------------------------------------------------------------------------

def analyze_storage(shard_dir: str | Path, strategy: str) -> StorageMetrics:
    """Compute per-sample storage footprint by reading shard headers."""
    shard_paths = sorted(Path(shard_dir).glob("shard_*.bin"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.bin files in {shard_dir}")

    metrics = StorageMetrics(strategy=strategy)

    for path in shard_paths:
        reader = ShardReader(str(path))
        h = reader.header

        is_uint8  = (h.version >= 2) and bool(h.flags & 0x01)
        has_dims  = (h.version >= 2) and bool(h.flags & 0x02)

        for i in range(reader.sample_count):
            sample = reader.read_sample(i)

            # Image dimensions
            if has_dims and sample.actual_height is not None:
                img_h, img_w = sample.actual_height, sample.actual_width
            else:
                img_h, img_w = h.image_height, h.image_width

            # Image bytes on disk
            dtype_sz = ELEM_SIZE_UINT8 if is_uint8 else ELEM_SIZE_F32
            img_bytes = NUM_CHANNELS * img_h * img_w * dtype_sz

            # Text bytes on disk (always fixed max_length in the shard format)
            text_bytes = NUM_TOKEN_ARRAYS * h.token_length * ELEM_SIZE_INT32

            # Per-sample overhead: v2 dimension prefix (8 bytes) + metadata (~64 bytes)
            overhead = (8 if has_dims else 0) + 64

            metrics.image_bytes_per_sample.append(img_bytes)
            metrics.text_bytes_per_sample.append(text_bytes)
            metrics.total_image_bytes += img_bytes
            metrics.total_text_bytes  += text_bytes
            metrics.total_overhead_bytes += overhead
            metrics.num_samples += 1

        reader.close()

    return metrics


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------

def simulate_batches(
    shard_dir: str | Path,
    strategy: str,
    batch_size: int,
    num_batches: int,
    warmup: int = 3,
) -> BatchMetrics:
    """Simulate batch reads from shard files and collect memory/latency metrics.

    Parameters
    ----------
    shard_dir : str | Path
        Directory containing the shard_*.bin files.
    strategy : str
        ``"static"`` or ``"dynamic"`` — controls how batch dimensions
        are computed.
    batch_size : int
        Number of samples per batch.
    num_batches : int
        Number of timed batches to measure.
    warmup : int
        Number of warm-up batches (not timed).
    """
    shard_paths = sorted(Path(shard_dir).glob("shard_*.bin"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.bin files in {shard_dir}")

    readers: list[ShardReader] = [ShardReader(str(p)) for p in shard_paths]
    sample_refs: list[tuple[int, int]] = []
    for ri, reader in enumerate(readers):
        for si in range(reader.sample_count):
            sample_refs.append((ri, si))

    if not sample_refs:
        for r in readers:
            r.close()
        return BatchMetrics(strategy=strategy, batch_size=batch_size)

    total = len(sample_refs)
    needed = (warmup + num_batches) * batch_size
    indices = list(range(total))
    if needed > total:
        indices = (indices * ((needed // total) + 2))[:needed]

    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    is_dynamic = (strategy == "dynamic")
    metrics = BatchMetrics(strategy=strategy, batch_size=batch_size)

    # Warm-up passes (cache priming)
    for b in range(warmup):
        for idx in indices[b * batch_size : (b + 1) * batch_size]:
            ri, si = sample_refs[idx % total]
            _ = readers[ri].read_sample(si)

    # Timed measurement passes
    offset = warmup * batch_size
    t_total_start = time.perf_counter()

    for b in range(num_batches):
        batch_slice = indices[offset + b * batch_size : offset + (b + 1) * batch_size]
        t_batch = time.perf_counter()

        batch_samples = []
        for idx in batch_slice:
            ri, si = sample_refs[idx % total]
            batch_samples.append(readers[ri].read_sample(si))

        latency_ms = (time.perf_counter() - t_batch) * 1000.0
        metrics.batch_latencies_ms.append(latency_ms)
        metrics.total_samples += len(batch_samples)
        metrics.num_batches += 1

        # ── Per-batch image dimensions ──────────────────────────────────────
        h0 = readers[0].header
        if is_dynamic:
            heights = []
            widths = []
            for s in batch_samples:
                if s.actual_height is not None:
                    heights.append(s.actual_height)
                    widths.append(s.actual_width)
                else:
                    heights.append(h0.image_height)
                    widths.append(h0.image_width)
            max_h = max(heights) if heights else h0.image_height
            max_w = max(widths) if widths else h0.image_width
        else:
            max_h = h0.image_height
            max_w = h0.image_width

        metrics.batch_img_heights.append(max_h)
        metrics.batch_img_widths.append(max_w)

        # GPU image tensor memory: N × C × H × W × float32 bytes
        # (always float32 after loader normalization, regardless of on-disk dtype)
        img_mem = batch_size * NUM_CHANNELS * max_h * max_w * ELEM_SIZE_F32
        metrics.image_memory_bytes.append(img_mem)

        # ── Per-batch text dimensions ───────────────────────────────────────
        if is_dynamic:
            max_q_len = max_a_len = 0
            for s in batch_samples:
                q_actual = int(np.sum(s.question_mask))
                a_actual = int(np.sum(s.answer_mask))
                max_q_len = max(max_q_len, q_actual)
                max_a_len = max(max_a_len, a_actual)
            effective_seq = max(max_q_len, max_a_len)
        else:
            effective_seq = h0.token_length

        metrics.batch_max_seq_lens.append(effective_seq)
        # GPU text tensor memory: N × seq_len × int32 × 4 arrays
        text_mem = batch_size * effective_seq * ELEM_SIZE_INT32 * NUM_TOKEN_ARRAYS
        metrics.text_memory_bytes.append(text_mem)

    metrics.total_time_s = time.perf_counter() - t_total_start

    for r in readers:
        r.close()

    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _pct_reduction(baseline: float, improved: float) -> float:
    if baseline <= 0:
        return 0.0
    return (1.0 - improved / baseline) * 100.0


def _savings_annotation(
    ax: Any,
    x_pos: float,
    y_ref: float,
    y_target: float,
    pct: float,
    abs_saved: float,
    unit: str = "MB",
) -> None:
    """Draw a savings arrow + label from static bar top to dynamic bar top."""
    ax.annotate(
        f"↓ {pct:.1f}% saved\n({abs_saved:.1f} {unit} reduction)",
        xy=(x_pos, y_target),
        xytext=(x_pos + 0.45, (y_ref + y_target) / 2),
        fontsize=11, fontweight="bold", color=COLOR_SAVINGS,
        ha="left", va="center",
        arrowprops=dict(
            arrowstyle="->", color=COLOR_SAVINGS, lw=1.8,
            connectionstyle="arc3,rad=0.1",
        ),
    )


# ---------------------------------------------------------------------------
# Individual plot functions — one per metric
# ---------------------------------------------------------------------------

def plot_storage_footprint(
    static_m: StorageMetrics,
    dynamic_m: StorageMetrics,
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Plot: On-disk storage footprint — stacked bar (image + text).
    Output: static_vs_dynamic_storage.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.patches as mpatches  # noqa: PLC0415

    apply_bright_style()

    s_img = static_m.image_mb
    s_txt = static_m.text_mb
    d_img = dynamic_m.image_mb
    d_txt = dynamic_m.text_mb

    x = np.array([0, 1])
    w = 0.45
    x_labels = [
        f"Static (v1)\nfloat32 · {STATIC_IMG_H}×{STATIC_IMG_W} fixed",
        f"Dynamic (v2)\nuint8 · AR-preserve ≤{DYNAMIC_MAX_DIM}px",
    ]

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        "On-Disk Storage Footprint: Static vs. Dynamic Padding",
        fontsize=19, fontweight="bold", y=1.01,
    )
    ax.set_title(
        f"Real dataset: {dataset_name}  •  {static_m.num_samples} samples",
        fontsize=13, color="#555555", pad=10,
    )

    b_img = ax.bar(x, [s_img, d_img], w, color=COLOR_IMAGE, alpha=0.92,
                   edgecolor="white", linewidth=1.5, label="Image data")
    b_txt = ax.bar(x, [s_txt, d_txt], w, bottom=[s_img, d_img],
                   color=COLOR_TEXT, alpha=0.92, edgecolor="white",
                   linewidth=1.5, label="Text data (tokens)")

    # Total labels on top of each bar
    for xi, (img_v, txt_v) in enumerate([(s_img, s_txt), (d_img, d_txt)]):
        total = img_v + txt_v
        ax.text(xi, total + max(s_img+s_txt, d_img+d_txt) * 0.02,
                f"{total:.1f} MB", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#1A252F")

    # Savings annotation
    total_s = s_img + s_txt
    total_d = d_img + d_txt
    pct = _pct_reduction(total_s, total_d)
    _savings_annotation(ax, 0.5, total_s, total_d, pct, total_s - total_d)

    ax.set_ylabel("Storage (MB)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=13)
    ax.set_ylim(0, max(total_s, total_d) * 1.40)

    # ── Legend outside plot area ──────────────────────────────────────────────
    img_patch  = mpatches.Patch(color=COLOR_IMAGE, alpha=0.92, label="Image data")
    txt_patch  = mpatches.Patch(color=COLOR_TEXT, alpha=0.92, label="Text data (tokens)")
    save_patch = mpatches.Patch(color=COLOR_SAVINGS, alpha=0.85,
                                label=f"Savings: ↓{pct:.1f}%")
    ax.legend(
        handles=[img_patch, txt_patch, save_patch],
        bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, fontsize=12,
        title="Data component", title_fontsize=12,
    )

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_batch_memory(
    static_b: BatchMetrics,
    dynamic_b: BatchMetrics,
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Plot: Per-batch GPU memory consumption (MB) — stacked image + text.
    Output: static_vs_dynamic_batch_memory.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.patches as mpatches  # noqa: PLC0415

    apply_bright_style()

    s_img = static_b.avg_image_mem_mb
    s_txt = static_b.avg_text_mem_mb
    d_img = dynamic_b.avg_image_mem_mb
    d_txt = dynamic_b.avg_text_mem_mb

    x = np.array([0, 1])
    w = 0.45
    x_labels = [
        f"Static\nfixed {STATIC_IMG_H}×{STATIC_IMG_W}, max_seq={MAX_SEQ_LENGTH}",
        f"Dynamic\nbatch max(H×W), batch max(seq_len)",
    ]

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        "Per-Batch GPU Memory: Static vs. Dynamic Padding",
        fontsize=19, fontweight="bold", y=1.01,
    )
    ax.set_title(
        f"Real dataset: {dataset_name}  •  batch_size={static_b.batch_size}  •  "
        f"avg over {static_b.num_batches} batches",
        fontsize=13, color="#555555", pad=10,
    )

    ax.bar(x, [s_img, d_img], w, color=COLOR_IMAGE, alpha=0.92,
           edgecolor="white", linewidth=1.5, label="Image tensor (float32)")
    ax.bar(x, [s_txt, d_txt], w, bottom=[s_img, d_img], color=COLOR_TEXT,
           alpha=0.92, edgecolor="white", linewidth=1.5,
           label="Token arrays (int32 × 4)")

    total_s = s_img + s_txt
    total_d = d_img + d_txt
    for xi, total in enumerate([total_s, total_d]):
        ax.text(xi, total + max(total_s, total_d) * 0.02,
                f"{total:.2f} MB", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#1A252F")

    pct = _pct_reduction(total_s, total_d)
    _savings_annotation(ax, 0.5, total_s, total_d, pct, total_s - total_d)

    ax.set_ylabel("GPU Memory per Batch (MB)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=13)
    ax.set_ylim(0, max(total_s, total_d) * 1.42)

    # ── Legend outside plot area ──────────────────────────────────────────────
    import matplotlib.patches as mpatches  # noqa: PLC0415
    patches = [
        mpatches.Patch(color=COLOR_IMAGE, alpha=0.92, label="Image tensor (float32)"),
        mpatches.Patch(color=COLOR_TEXT,  alpha=0.92, label="Token arrays (int32 × 4)"),
        mpatches.Patch(color=COLOR_SAVINGS, alpha=0.85, label=f"Savings: ↓{pct:.1f}%"),
    ]
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, fontsize=12,
        title="Memory component", title_fontsize=12,
    )

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_vit_flops(
    static_b: BatchMetrics,
    dynamic_b: BatchMetrics,
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Plot: ViT FLOPs proxy — average effective H×W per batch.
    ViT attention complexity is proportional to (H×W / patch_size²).
    Output: static_vs_dynamic_vit_flops.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.patches as mpatches  # noqa: PLC0415

    apply_bright_style()

    s_hw = static_b.avg_vit_hw
    d_hw = dynamic_b.avg_vit_hw
    pct  = _pct_reduction(s_hw, d_hw)

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        "ViT FLOPs Savings: Effective H×W per Batch",
        fontsize=19, fontweight="bold", y=1.01,
    )
    ax.set_title(
        f"Real dataset: {dataset_name}  •  FLOPs ∝ H×W  "
        f"(lower = faster ViT forward pass)",
        fontsize=13, color="#555555", pad=10,
    )

    x_labels = [
        f"Static\n{STATIC_IMG_H}×{STATIC_IMG_W} always",
        f"Dynamic\nbatch max(H)×max(W)",
    ]
    vals = [s_hw / 1_000, d_hw / 1_000]
    colors = [COLOR_STATIC, COLOR_DYNAMIC]

    bars = ax.bar([0, 1], vals, 0.45, color=colors, alpha=0.92,
                  edgecolor="white", linewidth=1.5)
    add_bar_labels(ax, bars, fmt="{:.1f}K")

    # FLOPs reduction annotation
    pct_abs = (s_hw - d_hw) / 1_000
    _savings_annotation(ax, 0.5, vals[0], vals[1], pct, pct_abs, unit="K H×W")

    ax.set_ylabel("Average Effective H×W (×10³)", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(x_labels, fontsize=13)
    ax.set_ylim(0, max(vals) * 1.45)

    # ── Legend outside plot area ──────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=COLOR_STATIC,  alpha=0.92, label=f"Static  — {s_hw:.0f} px²"),
        mpatches.Patch(color=COLOR_DYNAMIC, alpha=0.92,
                       label=f"Dynamic — {d_hw:.0f} px²  (↓{pct:.1f}%)"),
        mpatches.Patch(color=COLOR_SAVINGS, alpha=0.85,
                       label=f"FLOPs saved ≈ {pct:.1f}% of ViT compute"),
    ]
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, fontsize=12,
        title="Padding strategy", title_fontsize=12,
    )

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_llm_flops(
    static_b: BatchMetrics,
    dynamic_b: BatchMetrics,
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Plot: LLM FLOPs proxy — average effective sequence length per batch.
    Transformer attention is O(seq_len²).
    Output: static_vs_dynamic_llm_flops.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.patches as mpatches  # noqa: PLC0415

    apply_bright_style()

    s_seq = static_b.avg_seq_len
    d_seq = dynamic_b.avg_seq_len
    pct   = _pct_reduction(s_seq, d_seq)

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        "LLM FLOPs Savings: Effective Sequence Length per Batch",
        fontsize=19, fontweight="bold", y=1.01,
    )
    ax.set_title(
        f"Real dataset: {dataset_name}  •  FLOPs ∝ seq_len²  "
        f"(lower = less padding overhead in LLM layers)",
        fontsize=13, color="#555555", pad=10,
    )

    x_labels = [
        f"Static\nmax_length={MAX_SEQ_LENGTH} always",
        f"Dynamic\nbatch max(actual_len)",
    ]
    vals = [s_seq, d_seq]
    colors = [COLOR_STATIC, COLOR_DYNAMIC]

    bars = ax.bar([0, 1], vals, 0.45, color=colors, alpha=0.92,
                  edgecolor="white", linewidth=1.5)
    add_bar_labels(ax, bars, fmt="{:.0f} tokens")

    _savings_annotation(ax, 0.5, vals[0], vals[1], pct, s_seq - d_seq, unit="tokens")

    ax.set_ylabel("Average Effective Sequence Length (tokens)", fontsize=14,
                  fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(x_labels, fontsize=13)
    ax.set_ylim(0, max(vals) * 1.45)

    # ── Legend outside plot area ──────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=COLOR_STATIC,  alpha=0.92,
                       label=f"Static  — {s_seq:.0f} tokens (always)"),
        mpatches.Patch(color=COLOR_DYNAMIC, alpha=0.92,
                       label=f"Dynamic — {d_seq:.0f} tokens avg  (↓{pct:.1f}%)"),
        mpatches.Patch(color=COLOR_SAVINGS, alpha=0.85,
                       label=f"Attention FLOPs saved ≈ {pct:.1f}% (∝ seq_len²)"),
    ]
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, fontsize=12,
        title="Padding strategy", title_fontsize=12,
    )

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_throughput(
    static_b: BatchMetrics,
    dynamic_b: BatchMetrics,
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Plot: Shard read throughput (samples/sec) — ShardReader simulation.
    Output: static_vs_dynamic_throughput.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.patches as mpatches  # noqa: PLC0415

    apply_bright_style()

    s_tput = static_b.throughput_sps
    d_tput = dynamic_b.throughput_sps
    delta_pct = ((d_tput / s_tput) - 1.0) * 100.0 if s_tput > 0 else 0.0

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        "Shard Read Throughput: Static vs. Dynamic Padding",
        fontsize=19, fontweight="bold", y=1.01,
    )
    ax.set_title(
        f"Real dataset: {dataset_name}  •  batch_size={static_b.batch_size}  •  "
        f"measures end-to-end ShardReader read speed",
        fontsize=13, color="#555555", pad=10,
    )

    vals   = [s_tput, d_tput]
    colors = [COLOR_STATIC, COLOR_DYNAMIC]
    labels = ["Static (v1)", "Dynamic (v2)"]

    bars = ax.bar([0, 1], vals, 0.45, color=colors, alpha=0.92,
                  edgecolor="white", linewidth=1.5)
    add_bar_labels(ax, bars, fmt="{:.0f}")

    # Delta annotation
    sign = "↑" if delta_pct >= 0 else "↓"
    ax.annotate(
        f"{sign} {abs(delta_pct):.1f}% throughput difference",
        xy=(1, d_tput),
        xytext=(1.2, (s_tput + d_tput) / 2),
        fontsize=12, fontweight="bold",
        color=COLOR_SAVINGS if delta_pct >= 0 else COLOR_STATIC,
        arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5),
        ha="left",
    )

    ax.set_ylabel("Throughput (samples / second)", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, max(vals) * 1.35)

    # ── Legend outside plot area ──────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=COLOR_STATIC,  alpha=0.92,
                       label=f"Static  — {s_tput:.0f} samples/s"),
        mpatches.Patch(color=COLOR_DYNAMIC, alpha=0.92,
                       label=f"Dynamic — {d_tput:.0f} samples/s"),
    ]
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, fontsize=12,
        title="Padding strategy", title_fontsize=12,
    )

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_latency(
    static_b: BatchMetrics,
    dynamic_b: BatchMetrics,
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Plot: Per-batch read latency distribution — box plot with P50/P99 markers.
    Output: static_vs_dynamic_latency.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415
    import matplotlib.patches as mpatches  # noqa: PLC0415

    apply_bright_style()

    s_lat = static_b.batch_latencies_ms
    d_lat = dynamic_b.batch_latencies_ms

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        "Per-Batch Read Latency Distribution: Static vs. Dynamic Padding",
        fontsize=19, fontweight="bold", y=1.01,
    )
    ax.set_title(
        f"Real dataset: {dataset_name}  •  {static_b.num_batches} batches measured  •  "
        f"batch_size={static_b.batch_size}",
        fontsize=13, color="#555555", pad=10,
    )

    bp = ax.boxplot(
        [s_lat, d_lat],
        labels=["Static (v1)", "Dynamic (v2)"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="#1A252F", linewidth=2.5),
        whiskerprops=dict(color="#888888", linewidth=1.5, linestyle="--"),
        capprops=dict(color="#888888", linewidth=1.5),
        flierprops=dict(
            marker="o", markeredgecolor="#CCCCCC",
            markerfacecolor="#EEEEEE", markersize=4, alpha=0.6,
        ),
    )
    bp["boxes"][0].set_facecolor(COLOR_STATIC)
    bp["boxes"][0].set_alpha(0.80)
    bp["boxes"][1].set_facecolor(COLOR_DYNAMIC)
    bp["boxes"][1].set_alpha(0.80)

    # P50 / P99 text annotations
    for i, (bm, pos) in enumerate([(static_b, 1), (dynamic_b, 2)]):
        max_val = max(bm.batch_latencies_ms) if bm.batch_latencies_ms else 0
        ax.text(
            pos, max_val * 1.05,
            f"P50={bm.p50_latency_ms:.1f} ms\nP99={bm.p99_latency_ms:.1f} ms",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=COLOR_STATIC if i == 0 else COLOR_DYNAMIC,
        )

    ax.set_ylabel("Batch Read Latency (ms)", fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)

    # ── Legend outside plot area ──────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=COLOR_STATIC,  alpha=0.80,
                       label=f"Static  — P50={static_b.p50_latency_ms:.1f} ms, "
                             f"P99={static_b.p99_latency_ms:.1f} ms"),
        mpatches.Patch(color=COLOR_DYNAMIC, alpha=0.80,
                       label=f"Dynamic — P50={dynamic_b.p50_latency_ms:.1f} ms, "
                             f"P99={dynamic_b.p99_latency_ms:.1f} ms"),
        Line2D([0], [0], color="#1A252F", lw=2.5, label="Median (box centre line)"),
    ]
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, fontsize=12,
        title="Padding strategy", title_fontsize=12,
    )

    fig.tight_layout()
    save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    static_storage: StorageMetrics,
    dynamic_storage: StorageMetrics,
    static_batch: BatchMetrics,
    dynamic_batch: BatchMetrics,
) -> None:
    """Print a formatted comparison table to stdout."""
    print()
    print("═" * 88)
    print("  MULTIMODAL PADDING BENCHMARK: Static vs. Dynamic  (Real Data)")
    print("═" * 88)

    def row(metric: str, sv: str, dv: str, savings: str) -> None:
        print(f"  {metric:<28}  {sv:>18}  {dv:>18}  {savings:>14}")

    row("Metric", "Static (v1)", "Dynamic (v2)", "Savings / Δ")
    print("  " + "─" * 84)

    # Storage
    s_total = static_storage.total_mb
    d_total = dynamic_storage.total_mb
    pct     = _pct_reduction(s_total, d_total)
    row("Total storage (MB)",
        f"{s_total:.1f} MB", f"{d_total:.1f} MB", f"↓{pct:.1f}%")
    row("  └─ image (MB)",
        f"{static_storage.image_mb:.1f} MB", f"{dynamic_storage.image_mb:.1f} MB",
        f"↓{_pct_reduction(static_storage.image_mb, dynamic_storage.image_mb):.1f}%")
    row("  └─ text (MB)",
        f"{static_storage.text_mb:.1f} MB", f"{dynamic_storage.text_mb:.1f} MB",
        "(same format)")

    # Batch memory
    s_bm = static_batch.avg_total_mem_mb
    d_bm = dynamic_batch.avg_total_mem_mb
    row("Batch mem avg (MB)",
        f"{s_bm:.2f} MB", f"{d_bm:.2f} MB",
        f"↓{_pct_reduction(s_bm, d_bm):.1f}%")

    # ViT FLOPs
    s_hw = static_batch.avg_vit_hw
    d_hw = dynamic_batch.avg_vit_hw
    row("ViT H×W (avg)",
        f"{s_hw:.0f} px²", f"{d_hw:.0f} px²",
        f"↓{_pct_reduction(s_hw, d_hw):.1f}%")

    # LLM FLOPs
    s_sl = static_batch.avg_seq_len
    d_sl = dynamic_batch.avg_seq_len
    row("LLM seq_len (avg)",
        f"{s_sl:.0f} tok", f"{d_sl:.0f} tok",
        f"↓{_pct_reduction(s_sl, d_sl):.1f}%")

    # Throughput
    s_tp = static_batch.throughput_sps
    d_tp = dynamic_batch.throughput_sps
    delta = ((d_tp / s_tp) - 1.0) * 100.0 if s_tp > 0 else 0.0
    sign = "↑" if delta >= 0 else "↓"
    row("Read throughput (smp/s)",
        f"{s_tp:.0f}", f"{d_tp:.0f}",
        f"{sign}{abs(delta):.1f}%")

    # Latency
    row("Avg batch latency (ms)",
        f"{static_batch.avg_latency_ms:.2f}", f"{dynamic_batch.avg_latency_ms:.2f}", "")
    row("P99 batch latency (ms)",
        f"{static_batch.p99_latency_ms:.2f}", f"{dynamic_batch.p99_latency_ms:.2f}", "")

    print("═" * 88)
    print()


# ---------------------------------------------------------------------------
# Main benchmark orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    num_samples: int = 500,
    batch_size: int = 16,
    num_batches: int = 50,
    warmup: int = 5,
    static_shard_dir: str | Path | None = None,
    dynamic_shard_dir: str | Path | None = None,
    output_dir: Path | str = "benchmarks/results",
) -> None:
    """Run the complete Static vs. Dynamic padding benchmark suite.

    Parameters
    ----------
    num_samples : int
        Number of real samples to use (if not providing pre-built shards).
    batch_size : int
        Batch size for simulation.
    num_batches : int
        Number of timed batches to measure per strategy.
    warmup : int
        Number of warm-up batches (not timed).
    static_shard_dir : str | Path | None
        If provided, use pre-built v1 shards instead of generating them.
    dynamic_shard_dir : str | Path | None
        If provided, use pre-built v2 shards instead of generating them.
    output_dir : str | Path
        Directory for output PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║   Static vs. Dynamic Padding Benchmark — REAL DATA            ║")
    print("╠═══════════════════════════════════════════════════════════════╣")
    print(f"║  Samples:      {num_samples:<47}║")
    print(f"║  Batch size:   {batch_size:<47}║")
    print(f"║  Batches:      {num_batches:<47}║")
    print(f"║  Output:       {str(output_dir):<47}║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    tmp_root = output_dir / "_padding_tmp"
    own_shards = False

    if static_shard_dir is None or dynamic_shard_dir is None:
        # ── Generate shards from real data ────────────────────────────────────
        print(f"Loading {num_samples} real samples from HuggingFace ...")
        samples = load_real_dataset(num_samples, seed=0)
        dataset_name = samples[0].dataset_name if samples else "Unknown"
        print(f"  Dataset: {dataset_name}")
        print()

        print("Initialising tokenizer ...")
        tok_cfg = TokenizerConfig(
            model_name_or_path=_TOKENIZER_ID,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            trust_remote_code=False,
            add_special_tokens=True,
        )
        tokenizer = TextTokenizer(tok_cfg)
        print("  ✓ Tokenizer ready\n")

        static_shard_dir  = tmp_root / "v1_static"
        dynamic_shard_dir = tmp_root / "v2_dynamic"
        own_shards = True

        print("Writing static (v1) shards ...")
        write_static_shards(samples, Path(static_shard_dir), tokenizer)
        print()

        print("Writing dynamic (v2) shards ...")
        write_dynamic_shards(samples, Path(dynamic_shard_dir), tokenizer)
        print()
    else:
        # Use provided shard directories
        dataset_name = "external"
        print(f"Using pre-built shards:")
        print(f"  Static:  {static_shard_dir}")
        print(f"  Dynamic: {dynamic_shard_dir}")
        print()

    # ── Storage analysis ──────────────────────────────────────────────────────
    print("─" * 60)
    print("Analysing on-disk storage footprint ...")
    print("─" * 60)
    static_storage  = analyze_storage(static_shard_dir,  "static")
    dynamic_storage = analyze_storage(dynamic_shard_dir, "dynamic")
    print(f"  Static:  {static_storage.total_mb:.1f} MB  "
          f"(image={static_storage.image_mb:.1f} MB, "
          f"text={static_storage.text_mb:.1f} MB)")
    print(f"  Dynamic: {dynamic_storage.total_mb:.1f} MB  "
          f"(image={dynamic_storage.image_mb:.1f} MB, "
          f"text={dynamic_storage.text_mb:.1f} MB)")
    print()

    # ── Batch simulation ──────────────────────────────────────────────────────
    print("─" * 60)
    print("Simulating batch reads ...")
    print("─" * 60)

    print(f"  Static  — {num_batches} batches of size {batch_size} ...")
    static_batch = simulate_batches(
        static_shard_dir, "static", batch_size, num_batches, warmup,
    )
    print(f"    → {static_batch.throughput_sps:.1f} samples/s, "
          f"avg_latency={static_batch.avg_latency_ms:.2f} ms")

    print(f"  Dynamic — {num_batches} batches of size {batch_size} ...")
    dynamic_batch = simulate_batches(
        dynamic_shard_dir, "dynamic", batch_size, num_batches, warmup,
    )
    print(f"    → {dynamic_batch.throughput_sps:.1f} samples/s, "
          f"avg_latency={dynamic_batch.avg_latency_ms:.2f} ms")
    print()

    # ── Console summary ───────────────────────────────────────────────────────
    print_summary(static_storage, dynamic_storage, static_batch, dynamic_batch)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("─" * 60)
    print("Generating plots ...")
    print("─" * 60)

    try:
        plot_storage_footprint(
            static_storage, dynamic_storage, dataset_name,
            output_dir / "static_vs_dynamic_storage.png",
        )
        plot_batch_memory(
            static_batch, dynamic_batch, dataset_name,
            output_dir / "static_vs_dynamic_batch_memory.png",
        )
        plot_vit_flops(
            static_batch, dynamic_batch, dataset_name,
            output_dir / "static_vs_dynamic_vit_flops.png",
        )
        plot_llm_flops(
            static_batch, dynamic_batch, dataset_name,
            output_dir / "static_vs_dynamic_llm_flops.png",
        )
        plot_throughput(
            static_batch, dynamic_batch, dataset_name,
            output_dir / "static_vs_dynamic_throughput.png",
        )
        plot_latency(
            static_batch, dynamic_batch, dataset_name,
            output_dir / "static_vs_dynamic_latency.png",
        )
    except Exception as plot_exc:
        print(f"  ⚠  Plot generation error: {plot_exc}")
        import traceback  # noqa: PLC0415
        traceback.print_exc()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if own_shards and tmp_root.exists():
        print()
        print("Cleaning up temporary shard files ...")
        shutil.rmtree(tmp_root, ignore_errors=True)
        print("  ✓ Done")

    print()
    print(f"Benchmark complete!  6 plots saved to: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "VLM Padding Benchmark — Static vs. Dynamic (Real HuggingFace Data)\n"
            "Measures storage, GPU memory, ViT/LLM FLOPs, and read latency.\n"
            "Outputs one independent PNG per metric in benchmarks/results/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--num-samples", type=int, default=500,
        help="Number of real samples to load (default: 500).",
    )
    data_group.add_argument(
        "--static-shards", type=str, default=None,
        help="Path to pre-built v1 (static) shard directory.",
    )
    parser.add_argument(
        "--dynamic-shards", type=str, default=None,
        help="Path to pre-built v2 (dynamic) shard directory "
             "(required when --static-shards is provided).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for simulation (default: 16).",
    )
    parser.add_argument(
        "--num-batches", type=int, default=50,
        help="Number of timed batches per strategy (default: 50).",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warm-up batches (default: 5).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results",
        help="Output directory for PNGs (default: benchmarks/results).",
    )
    args = parser.parse_args()

    if args.static_shards and not args.dynamic_shards:
        parser.error("--dynamic-shards is required when --static-shards is provided.")

    run_benchmark(
        num_samples=getattr(args, "num_samples", 500),
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        warmup=args.warmup,
        static_shard_dir=args.static_shards,
        dynamic_shard_dir=args.dynamic_shards,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
