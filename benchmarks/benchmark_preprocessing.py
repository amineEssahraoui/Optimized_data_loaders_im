#!/usr/bin/env python3
"""
benchmark_preprocessing.py — Production-Grade Preprocessing Benchmark Suite
=============================================================================

Measures throughput (samples/s) and memory for the sequential vs.
streaming multiprocessing preprocessing pipelines.  All data is loaded
from a REAL public HuggingFace dataset — no synthetic data.

Benchmarks (each saved as a standalone PNG in benchmarks/results/):

  1. throughput_vs_workers.png
        Samples/sec for Sequential, Streaming 1-worker, 2-worker, 4-worker, …
        Includes a speedup overlay line on a twin axis.

  2. memory_vs_workers.png
        Peak RSS memory (MB, via tracemalloc) for each configuration.

  3. throughput_vs_dataset_size.png
        How sequential + best-worker-count scale as dataset grows
        (subset sizes: 500 → 1k → 2k → 5k samples).

  4. dataloader_throughput.png
        Batches/sec for our ShardReader (proxy for C++ AsyncShardLoader)
        vs. a naive PyTorch IterableDataset reading raw JPEG + tokenizing.

Usage::

    # Typical quick run (~2-3 min)
    python benchmarks/benchmark_preprocessing.py --num-samples 1000

    # Full production-scale run
    python benchmarks/benchmark_preprocessing.py \\
        --num-samples 5000 \\
        --workers 1 2 4 8 \\
        --output-dir benchmarks/results

    # Skip DataLoader benchmark (no torch installed)
    python benchmarks/benchmark_preprocessing.py --no-dataloader-bench
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import gc
import glob
import os
import shutil
import sys
import tempfile
import time
import tracemalloc
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
    COLOR_SPEEDUP,
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
from preprocessing.shard_writer import ShardWriter  # noqa: E402
from preprocessing.tokenizer import TextTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = _BENCH_DIR / "results"

# Lightweight tokenizer — bert-base-uncased is small and universally cached.
_TOKENIZER_ID = "bert-base-uncased"
_MAX_SEQ_LEN   = 128
_IMG_MAX_DIM   = 224   # keep images small so timing reflects I/O, not compute
_SHARD_SAMPLES = 200   # samples per shard (keeps individual shards manageable)


# ---------------------------------------------------------------------------
# Benchmark result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a single timed benchmark run."""
    label: str
    num_workers: int
    num_samples: int
    elapsed_seconds: float
    throughput_sps: float        # samples per second
    peak_memory_mb: float
    shards_written: int
    total_bytes: int
    errors: int

    def speedup_over(self, baseline_sps: float) -> float:
        if baseline_sps <= 0:
            return 1.0
        return self.throughput_sps / baseline_sps


@dataclass
class BenchmarkSuite:
    """Collection of all benchmark results."""
    worker_results: list[RunResult] = field(default_factory=list)
    scaling_results: list[RunResult] = field(default_factory=list)
    dataloader_results: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared pipeline config factory
# ---------------------------------------------------------------------------

def _make_config(
    output_dir: Path,
    num_workers: int,
    format_version: int = 2,
) -> PipelineConfig:
    """Build a lightweight PipelineConfig for benchmarking."""
    img_cfg = ImageConfig(
        max_image_dim=_IMG_MAX_DIM,
        storage_dtype="uint8",
        dynamic_padding=True,
        color_space="RGB",
        interpolation="bilinear",
    )
    tok_cfg = TokenizerConfig(
        model_name_or_path=_TOKENIZER_ID,
        max_length=_MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        trust_remote_code=False,
        add_special_tokens=True,
    )
    shard_cfg = ShardConfig(
        output_dir=str(output_dir),
        shard_size_mb=512,
        max_samples_per_shard=_SHARD_SAMPLES,
        compression=None,
        alignment_bytes=64,
        format_version=format_version,
    )
    return PipelineConfig(
        image=img_cfg,
        tokenizer=tok_cfg,
        shard=shard_cfg,
        num_workers=num_workers,
        shuffling="none",
        log_level="WARNING",
    )


# ---------------------------------------------------------------------------
# Sequential baseline runner
# ---------------------------------------------------------------------------

def _run_sequential(
    samples: list[VQASample],
    config: PipelineConfig,
    output_dir: Path,
) -> RunResult:
    """Run the sequential (single-threaded) preprocessing pipeline.

    This is the Phase 1-3 baseline: one tokenizer, one writer, no
    multiprocessing.  Used as the denominator for speedup calculations.
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    tokenizer = TextTokenizer(config.tokenizer)
    writer = ShardWriter(config.shard, config.image, config.tokenizer)
    shard_index = 0
    shard_path = output_dir / f"shard_{shard_index:04d}.bin"
    writer.open(shard_path)

    processed = errors = 0
    total_bytes = 0

    for sample in samples:
        try:
            img_arr, orig_h, orig_w, _ah, _aw = resize_preserve_aspect(
                sample.image_bytes, config.image
            )
            q_tok = tokenizer.tokenize(sample.question)
            a_tok = tokenizer.tokenize(sample.answer)

            meta = dict(sample.metadata)
            meta["sample_id"] = sample.sample_id

            writer.add_sample(
                image_tensor=img_arr,
                question_ids=q_tok.input_ids,
                question_mask=q_tok.attention_mask,
                answer_ids=a_tok.input_ids,
                answer_mask=a_tok.attention_mask,
                metadata=meta,
                orig_height=orig_h,
                orig_width=orig_w,
            )
            processed += 1

            if writer.should_rotate():
                total_bytes += writer.current_size_bytes
                writer.close()
                shard_index += 1
                shard_path = output_dir / f"shard_{shard_index:04d}.bin"
                writer = ShardWriter(config.shard, config.image, config.tokenizer)
                writer.open(shard_path)

        except Exception:
            errors += 1

    total_bytes += writer.current_size_bytes
    writer.close()

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return RunResult(
        label="Sequential (1 thread)",
        num_workers=0,
        num_samples=len(samples),
        elapsed_seconds=round(elapsed, 4),
        throughput_sps=round(processed / elapsed, 2) if elapsed > 0 else 0.0,
        peak_memory_mb=round(peak_bytes / (1024 ** 2), 2),
        shards_written=shard_index + 1,
        total_bytes=total_bytes,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Streaming multiprocessing runner
# ---------------------------------------------------------------------------

def _run_streaming(
    samples: list[VQASample],
    config: PipelineConfig,
    output_dir: Path,
    num_workers: int,
) -> RunResult:
    """Run the Phase 4 streaming multiprocessing pipeline.

    Wraps ``_run_streaming_pipeline`` with timing + tracemalloc.
    """
    from preprocessing.pipeline import _run_streaming_pipeline  # noqa: PLC0415

    bench_config = dc.replace(config, num_workers=num_workers)
    rng = np.random.RandomState(42)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    _last_idx, total_bytes, processed, errors = _run_streaming_pipeline(
        samples, bench_config, output_dir, rng,
    )

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    shard_count = len(list(output_dir.glob("shard_*.bin")))

    return RunResult(
        label=f"Streaming ({num_workers} worker{'s' if num_workers != 1 else ''})",
        num_workers=num_workers,
        num_samples=len(samples),
        elapsed_seconds=round(elapsed, 4),
        throughput_sps=round(processed / elapsed, 2) if elapsed > 0 else 0.0,
        peak_memory_mb=round(peak_bytes / (1024 ** 2), 2),
        shards_written=shard_count,
        total_bytes=total_bytes,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# DataLoader benchmark
# ---------------------------------------------------------------------------

def _bench_shard_reader(
    shard_dir: Path,
    batch_size: int = 16,
    num_batches: int = 30,
    warmup: int = 3,
) -> dict[str, Any]:
    """Benchmark C++ AsyncShardLoader directly."""
    try:
        import vlm_loader_py
    except ImportError:
        return {
            "label": "AsyncShardLoader\n(C++ extension not built)",
            "batches_per_sec": 0.0,
            "samples_per_sec": 0.0,
            "elapsed": 0.0,
        }

    shard_paths = [str(p) for p in sorted(shard_dir.glob("shard_*.bin"))]
    if not shard_paths:
        return {"label": "AsyncShardLoader", "batches_per_sec": 0.0, "samples_per_sec": 0.0, "elapsed": 0.0}

    config = vlm_loader_py.AsyncLoaderConfig()
    config.shard_paths = shard_paths
    config.batch_size = batch_size
    config.prefetch_depth = 2
    config.num_workers = min(4, os.cpu_count() or 4)
    config.seed = 42
    config.shuffle_buffer_size = 0  # No shuffle for benchmarking raw throughput

    loader = vlm_loader_py.AsyncShardLoader(config)

    # Warmup
    for _ in range(warmup):
        try:
            _ = loader.next()
        except StopIteration:
            loader.reset()
            _ = loader.next()

    # Timed
    t0 = time.perf_counter()
    for _ in range(num_batches):
        try:
            _ = loader.next()
        except StopIteration:
            loader.reset()
            _ = loader.next()
    elapsed = time.perf_counter() - t0

    batches_ps = num_batches / elapsed if elapsed > 0 else 0.0
    samples_ps = (num_batches * batch_size) / elapsed if elapsed > 0 else 0.0
    return {
        "label": "C++ AsyncShardLoader\n(vlm_loader_py)",
        "batches_per_sec": round(batches_ps, 2),
        "samples_per_sec": round(samples_ps, 2),
        "elapsed": round(elapsed, 3),
    }


def _bench_pytorch_dataloader(
    samples: list[VQASample],
    batch_size: int = 16,
    num_batches: int = 30,
    warmup: int = 3,
) -> dict[str, Any]:
    """Benchmark a naive PyTorch IterableDataset + DataLoader.

    Each iteration: decode JPEG with PIL, resize, convert to tensor.
    No tokenisation — simulates a minimal image-only baseline.
    """
    try:
        import torch  # noqa: PLC0415
        from torch.utils.data import DataLoader, IterableDataset  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        import io as _io  # noqa: PLC0415
    except ImportError:
        return {
            "label": "PyTorch DataLoader\n(not installed)",
            "batches_per_sec": 0.0,
            "samples_per_sec": 0.0,
        }

    class _RawDataset(IterableDataset):  # type: ignore[type-arg]
        def __init__(self, vqa_samples: list[VQASample]) -> None:
            self._samples = vqa_samples

        def __iter__(self):  # type: ignore[override]
            for s in self._samples:
                img = Image.open(_io.BytesIO(s.image_bytes)).convert("RGB")
                img = img.resize((_IMG_MAX_DIM, _IMG_MAX_DIM))
                arr = np.array(img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr.transpose(2, 0, 1))
                yield tensor

    needed = (warmup + num_batches) * batch_size
    repeat_samples = (samples * ((needed // len(samples)) + 2))[:needed]

    dataset = _RawDataset(repeat_samples)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    loader_iter = iter(loader)

    # Warmup
    for _ in range(warmup):
        try:
            _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)

    t0 = time.perf_counter()
    counted = 0
    for _ in range(num_batches):
        try:
            _ = next(loader_iter)
            counted += 1
        except StopIteration:
            break
    elapsed = time.perf_counter() - t0

    batches_ps = counted / elapsed if elapsed > 0 else 0.0
    samples_ps = (counted * batch_size) / elapsed if elapsed > 0 else 0.0
    return {
        "label": "PyTorch DataLoader\n(raw JPEG decode)",
        "batches_per_sec": round(batches_ps, 2),
        "samples_per_sec": round(samples_ps, 2),
        "elapsed": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Plotting — one metric, one file
# ---------------------------------------------------------------------------

def plot_throughput_vs_workers(
    results: list[RunResult],
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Plot 1: Throughput (samples/s) vs. configuration with speedup overlay.
    Output: throughput_vs_workers.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    apply_bright_style()

    labels      = [r.label for r in results]
    throughputs = [r.throughput_sps for r in results]
    baseline    = results[0].throughput_sps if results else 1.0
    speedups    = [t / baseline if baseline > 0 else 1.0 for t in throughputs]

    n = len(results)
    bar_colors = [BRIGHT_PALETTE[i % len(BRIGHT_PALETTE)] for i in range(n)]
    x = np.arange(n)
    bar_w = 0.55

    fig, ax1 = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        "Preprocessing Throughput: Sequential vs. Streaming Multiprocessing",
        fontsize=19, fontweight="bold", y=1.01,
    )
    ax1.set_title(
        f"Real dataset: {dataset_name}  •  {results[0].num_samples} samples",
        fontsize=13, color="#555555", pad=10,
    )

    bars = ax1.bar(x, throughputs, bar_w, color=bar_colors, edgecolor="white",
                   linewidth=1.5, zorder=3, alpha=0.92)
    add_bar_labels(ax1, bars, fmt="{:.1f}")

    ax1.set_ylabel("Throughput (samples / second)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12, rotation=15, ha="right")
    ax1.set_ylim(0, max(throughputs) * 1.22)

    # ── Speedup line on secondary axis ──────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(x, speedups, color=COLOR_SPEEDUP, marker="D", markersize=11,
             linewidth=2.5, linestyle="--", zorder=5)
    for xi, sp in zip(x, speedups):
        ax2.annotate(
            f"{sp:.2f}×",
            (xi, sp),
            textcoords="offset points", xytext=(0, 14),
            ha="center", fontsize=11, fontweight="bold", color=COLOR_SPEEDUP,
        )
    ax2.set_ylabel("Speedup over Sequential (×)", fontsize=13,
                   fontweight="bold", color=COLOR_SPEEDUP)
    ax2.tick_params(axis="y", colors=COLOR_SPEEDUP)
    ax2.spines["right"].set_edgecolor(COLOR_SPEEDUP)
    ax2.set_ylim(0, max(speedups) * 1.4)

    # ── Legend outside plot area ─────────────────────────────────────────────
    legend_handles = (
        [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.92)
         for c in bar_colors]
        + [Line2D([0], [0], color=COLOR_SPEEDUP, lw=2.5,
                  marker="D", markersize=8, linestyle="--",
                  label="Speedup (×)")]
    )
    legend_labels = labels + ["Speedup (×)"]
    ax1.legend(
        legend_handles, legend_labels,
        bbox_to_anchor=(1.18, 1), loc="upper left",
        borderaxespad=0, fontsize=12,
        title="Configuration", title_fontsize=12,
    )

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_memory_vs_workers(
    results: list[RunResult],
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Plot 2: Peak tracemalloc memory (MB) per configuration.
    Output: memory_vs_workers.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    apply_bright_style()

    labels    = [r.label for r in results]
    memory_mb = [r.peak_memory_mb for r in results]
    n = len(results)
    colors = [BRIGHT_PALETTE[(i + 2) % len(BRIGHT_PALETTE)] for i in range(n)]
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        "Peak Process Memory (tracemalloc) by Configuration",
        fontsize=19, fontweight="bold", y=1.01,
    )
    ax.set_title(
        f"Real dataset: {dataset_name}  •  {results[0].num_samples} samples",
        fontsize=13, color="#555555", pad=10,
    )

    bars = ax.bar(x, memory_mb, 0.55, color=colors, edgecolor="white",
                  linewidth=1.5, zorder=3, alpha=0.92)
    add_bar_labels(ax, bars, fmt="{:.1f} MB")

    # Baseline reference line
    if memory_mb:
        ax.axhline(y=memory_mb[0], color="#AAAAAA", linestyle=":",
                   linewidth=1.8, alpha=0.8, zorder=2,
                   label=f"Sequential baseline ({memory_mb[0]:.1f} MB)")

    ax.set_ylabel("Peak Memory (MB)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=15, ha="right")
    ax.set_ylim(0, max(memory_mb) * 1.25)

    # ── Legend outside plot area ─────────────────────────────────────────────
    import matplotlib.patches as mpatches  # noqa: PLC0415
    bar_patches = [mpatches.Patch(color=colors[i], alpha=0.92, label=labels[i])
                   for i in range(n)]
    from matplotlib.lines import Line2D  # noqa: PLC0415
    ref_line = Line2D([0], [0], color="#AAAAAA", linestyle=":", lw=1.8,
                      label=f"Sequential baseline ({memory_mb[0]:.1f} MB)")
    ax.legend(
        handles=bar_patches + [ref_line],
        bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, fontsize=12,
        title="Configuration", title_fontsize=12,
    )

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_throughput_vs_dataset_size(
    size_results: list[RunResult],
    output_path: Path,
) -> None:
    """
    Plot 3: Throughput vs. dataset size (scaling benchmark).
    Groups results by configuration, plots lines with markers.
    Output: throughput_vs_dataset_size.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    apply_bright_style()

    # Group by label
    label_map: dict[str, dict[str, list]] = {}
    for r in size_results:
        if r.label not in label_map:
            label_map[r.label] = {"sizes": [], "throughputs": []}
        label_map[r.label]["sizes"].append(r.num_samples)
        label_map[r.label]["throughputs"].append(r.throughput_sps)

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        "Throughput Scaling: How Preprocessing Scales with Dataset Size",
        fontsize=19, fontweight="bold", y=1.01,
    )
    ax.set_title(
        "Sequential vs. best multiprocessing configuration",
        fontsize=13, color="#555555", pad=10,
    )

    markers = ["o", "s", "D", "^", "P", "X"]
    lines = []
    for i, (label, data) in enumerate(label_map.items()):
        sizes = data["sizes"]
        tputs = data["throughputs"]
        sorted_pairs = sorted(zip(sizes, tputs))
        xs = [p[0] for p in sorted_pairs]
        ys = [p[1] for p in sorted_pairs]
        color = BRIGHT_PALETTE[i % len(BRIGHT_PALETTE)]
        line, = ax.plot(xs, ys, color=color, marker=markers[i % len(markers)],
                        markersize=11, linewidth=2.5, label=label)
        lines.append(line)
        for x_pt, y_pt in zip(xs, ys):
            ax.annotate(
                f"{y_pt:.0f}",
                (x_pt, y_pt),
                textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=10, fontweight="bold", color=color,
            )

    ax.set_xlabel("Dataset Size (number of samples)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Throughput (samples / second)", fontsize=14, fontweight="bold")

    all_sizes = sorted({r.num_samples for r in size_results})
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([f"{s:,}" for s in all_sizes], fontsize=12)
    ax.set_ylim(bottom=0)

    # ── Legend outside plot area ─────────────────────────────────────────────
    ax.legend(
        handles=lines,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, fontsize=12,
        title="Pipeline variant", title_fontsize=12,
    )

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_dataloader_throughput(
    dl_results: list[dict[str, Any]],
    batch_size: int,
    output_path: Path,
) -> None:
    """
    Plot 4: DataLoader throughput — ShardReader vs PyTorch DataLoader.
    Shows both batches/sec and samples/sec as grouped bars.
    Output: dataloader_throughput.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    apply_bright_style()

    labels  = [r["label"] for r in dl_results]
    bps     = [r["batches_per_sec"] for r in dl_results]
    sps     = [r["samples_per_sec"] for r in dl_results]

    n = len(dl_results)
    x = np.arange(n)
    w = 0.35
    colors_b = [BRIGHT_PALETTE[0], BRIGHT_PALETTE[1]]
    colors_s = [BRIGHT_PALETTE[2], BRIGHT_PALETTE[3]]

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        "DataLoader Throughput: ShardReader vs. PyTorch IterableDataset",
        fontsize=19, fontweight="bold", y=1.01,
    )
    ax.set_title(
        f"Batch size = {batch_size}  •  Measuring end-to-end read throughput",
        fontsize=13, color="#555555", pad=10,
    )

    bars_b = ax.bar(x - w / 2, bps, w, color=[colors_b[i % 2] for i in range(n)],
                    edgecolor="white", linewidth=1.5, alpha=0.92,
                    label="Batches / second")
    bars_s = ax.bar(x + w / 2, sps, w, color=[colors_s[i % 2] for i in range(n)],
                    edgecolor="white", linewidth=1.5, alpha=0.92,
                    label="Samples / second")

    add_bar_labels(ax, bars_b, fmt="{:.1f}")
    add_bar_labels(ax, bars_s, fmt="{:.0f}")

    ax.set_ylabel("Throughput", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, max(max(bps, default=0), max(sps, default=0)) * 1.28)

    # ── Legend outside plot area ─────────────────────────────────────────────
    import matplotlib.patches as mpatches  # noqa: PLC0415
    batches_patch  = mpatches.Patch(color=colors_b[0], alpha=0.92,
                                    label="Batches / second")
    samples_patch  = mpatches.Patch(color=colors_s[0], alpha=0.92,
                                    label="Samples / second")
    loader_patches = [
        mpatches.Patch(color=colors_b[i % 2], alpha=0.92, label=labels[i])
        for i in range(n)
    ]
    ax.legend(
        handles=[batches_patch, samples_patch] + loader_patches,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, fontsize=12,
        title="Metric / Loader", title_fontsize=12,
    )

    fig.tight_layout()
    save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(results: list[RunResult], title: str = "Worker Scaling") -> None:
    baseline_sps = results[0].throughput_sps if results else 1.0
    col = 32

    print()
    print("═" * 110)
    print(f"  {title}")
    print("═" * 110)
    hdr = (
        f"{'Configuration':<{col}} "
        f"{'Workers':>7} {'Samples':>8} {'Time(s)':>9} "
        f"{'Throughput':>12} {'Speedup':>9} {'Peak RAM':>10} {'Errors':>7}"
    )
    print(hdr)
    print("─" * 110)
    for r in results:
        sp = r.speedup_over(baseline_sps)
        print(
            f"{r.label:<{col}} "
            f"{r.num_workers:>7} {r.num_samples:>8} {r.elapsed_seconds:>9.3f} "
            f"{r.throughput_sps:>10.1f} s/s {sp:>7.2f}× "
            f"{r.peak_memory_mb:>7.1f} MB {r.errors:>7}"
        )
    print("═" * 110)
    if results:
        best = max(results, key=lambda r: r.throughput_sps)
        print(f"  Best: {best.label} → {best.throughput_sps:.1f} samples/s "
              f"({best.speedup_over(baseline_sps):.2f}× speedup)")
    print()


# ---------------------------------------------------------------------------
# Main benchmark orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    num_samples: int = 1000,
    worker_counts: list[int] | None = None,
    scaling_sizes: list[int] | None = None,
    output_dir: Path | str = "benchmarks/results",
    batch_size: int = 16,
    dataloader_batches: int = 40,
    run_dataloader_bench: bool = True,
) -> BenchmarkSuite:
    """Run the complete preprocessing benchmark suite on real data.

    Parameters
    ----------
    num_samples : int
        Number of real samples to load for the worker-scaling benchmark.
    worker_counts : list[int] | None
        Worker counts to test in the multiprocessing benchmark.
        Defaults to [1, 2, 4] clamped to available CPUs.
    scaling_sizes : list[int] | None
        Dataset sizes for the scaling benchmark.
        Defaults to [250, 500, 1000] (adjusted if num_samples is smaller).
    output_dir : Path | str
        Where to save PNGs.
    batch_size : int
        Batch size for the DataLoader throughput benchmark.
    dataloader_batches : int
        Number of measured batches for the DataLoader benchmark.
    run_dataloader_bench : bool
        Whether to run the DataLoader comparison benchmark.
    """
    if worker_counts is None:
        max_cpus = os.cpu_count() or 4
        worker_counts = [w for w in [1, 2, 4, 8] if w <= max_cpus]
        if not worker_counts:
            worker_counts = [1]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset scaling defaults (clamp to num_samples) ──────────────────────
    if scaling_sizes is None:
        candidates = [250, 500, 1000, 2000]
        scaling_sizes = [s for s in candidates if s <= num_samples]
        if not scaling_sizes:
            scaling_sizes = [num_samples]
        if num_samples not in scaling_sizes:
            scaling_sizes.append(num_samples)
        scaling_sizes = sorted(set(scaling_sizes))

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   VLM Preprocessing Benchmark Suite — REAL DATA             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Total samples:   {num_samples:<43}║")
    print(f"║  Worker counts:   {str(worker_counts):<43}║")
    print(f"║  Scaling sizes:   {str(scaling_sizes):<43}║")
    print(f"║  Output dir:      {str(output_dir):<43}║")
    print(f"║  CPU cores:       {os.cpu_count() or 'unknown':<43}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Step 1: Load real dataset ─────────────────────────────────────────────
    max_needed = max(num_samples, max(scaling_sizes))
    print(f"Loading {max_needed} real samples from HuggingFace ...")
    all_samples = load_real_dataset(max_needed, seed=42)
    dataset_name = all_samples[0].dataset_name if all_samples else "Unknown"
    print(f"  Dataset: {dataset_name}")
    print()

    # Use the first num_samples for the worker-scaling benchmark
    samples_for_workers = all_samples[:num_samples]
    suite = BenchmarkSuite()

    # ── Temporary shard directory ──────────────────────────────────────────────
    tmp_root = output_dir / "_bench_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    # ── Step 2: Worker-scaling benchmark ─────────────────────────────────────
    print("─" * 60)
    print("BENCHMARK 1 & 2: Throughput + Memory vs. Worker Count")
    print("─" * 60)

    # Sequential baseline
    seq_dir = tmp_root / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    seq_cfg = _make_config(seq_dir, num_workers=0)

    print("  Running sequential baseline ...")
    seq_result = _run_sequential(samples_for_workers, seq_cfg, seq_dir)
    suite.worker_results.append(seq_result)
    print(f"    → {seq_result.throughput_sps:.1f} samples/s, "
          f"{seq_result.peak_memory_mb:.1f} MB peak")

    # Streaming multiprocessing
    for nw in worker_counts:
        worker_dir = tmp_root / f"workers_{nw}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_cfg = _make_config(worker_dir, num_workers=nw)

        print(f"  Running streaming pipeline ({nw} worker(s)) ...")
        res = _run_streaming(samples_for_workers, worker_cfg, worker_dir, nw)
        suite.worker_results.append(res)
        print(f"    → {res.throughput_sps:.1f} samples/s, "
              f"{res.peak_memory_mb:.1f} MB peak")

    _print_summary(suite.worker_results, "Worker Scaling Results")

    # ── Step 3: Dataset scaling benchmark ─────────────────────────────────────
    print("─" * 60)
    print("BENCHMARK 3: Throughput vs. Dataset Size")
    print("─" * 60)

    best_worker = max(worker_counts) if worker_counts else 1
    print(f"  Running sequential + {best_worker}-worker at each dataset size ...")

    for size in scaling_sizes:
        subset = all_samples[:size]

        # Sequential at this size
        sc_seq_dir = tmp_root / f"scale_seq_{size}"
        sc_seq_dir.mkdir(parents=True, exist_ok=True)
        sc_seq_cfg = _make_config(sc_seq_dir, num_workers=0)
        r_seq = _run_sequential(subset, sc_seq_cfg, sc_seq_dir)
        r_seq = dc.replace(r_seq, num_samples=size)
        suite.scaling_results.append(r_seq)

        # Best multiprocessing at this size
        sc_mp_dir = tmp_root / f"scale_mp_{size}"
        sc_mp_dir.mkdir(parents=True, exist_ok=True)
        sc_mp_cfg = _make_config(sc_mp_dir, num_workers=best_worker)
        r_mp = _run_streaming(subset, sc_mp_cfg, sc_mp_dir, best_worker)
        r_mp = dc.replace(r_mp, num_samples=size)
        suite.scaling_results.append(r_mp)

        print(f"  Size {size:>5}: Sequential={r_seq.throughput_sps:.1f} s/s, "
              f"Streaming={r_mp.throughput_sps:.1f} s/s")

    # ── Step 4: DataLoader benchmark ──────────────────────────────────────────
    if run_dataloader_bench:
        print()
        print("─" * 60)
        print("BENCHMARK 4: DataLoader Throughput Comparison")
        print("─" * 60)

        # Write shards for the ShardReader benchmark (reuse sequential shards)
        shard_bench_dir = tmp_root / "seq"

        print("  Benchmarking ShardReader ...")
        sr_result = _bench_shard_reader(
            shard_bench_dir, batch_size, dataloader_batches
        )
        print(f"    → {sr_result['batches_per_sec']:.2f} batches/s, "
              f"{sr_result['samples_per_sec']:.1f} samples/s")

        print("  Benchmarking PyTorch DataLoader ...")
        pt_result = _bench_pytorch_dataloader(
            samples_for_workers, batch_size, dataloader_batches
        )
        print(f"    → {pt_result['batches_per_sec']:.2f} batches/s, "
              f"{pt_result['samples_per_sec']:.1f} samples/s")

        suite.dataloader_results = [sr_result, pt_result]

    # ── Step 5: Generate all plots ────────────────────────────────────────────
    print()
    print("─" * 60)
    print("Generating plots ...")
    print("─" * 60)

    try:
        plot_throughput_vs_workers(
            suite.worker_results,
            dataset_name,
            output_dir / "throughput_vs_workers.png",
        )
        plot_memory_vs_workers(
            suite.worker_results,
            dataset_name,
            output_dir / "memory_vs_workers.png",
        )
        plot_throughput_vs_dataset_size(
            suite.scaling_results,
            output_dir / "throughput_vs_dataset_size.png",
        )
        if suite.dataloader_results:
            plot_dataloader_throughput(
                suite.dataloader_results,
                batch_size,
                output_dir / "dataloader_throughput.png",
            )
    except Exception as plot_exc:
        print(f"  ⚠  Plot generation error: {plot_exc}")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    print()
    print("Cleaning up temporary shard files ...")
    shutil.rmtree(tmp_root, ignore_errors=True)
    print("  ✓ Cleanup complete")

    print()
    print(f"Benchmark complete!  Results saved to: {output_dir.resolve()}")
    return suite


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "VLM Preprocessing Benchmark — Real HuggingFace Data\n"
            "Benchmarks sequential vs streaming multiprocessing preprocessing\n"
            "on REAL VQA images.  Outputs one PNG per metric."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="Number of real samples to load (default: 1000).",
    )
    parser.add_argument(
        "--workers", type=int, nargs="+", default=None,
        help="Worker counts for multiprocessing benchmark (default: 1 2 4 8).",
    )
    parser.add_argument(
        "--scaling-sizes", type=int, nargs="+", default=None,
        help="Dataset sizes for the scaling benchmark (default: auto).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results",
        help="Directory for PNG outputs (default: benchmarks/results).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for DataLoader benchmark (default: 16).",
    )
    parser.add_argument(
        "--dataloader-batches", type=int, default=40,
        help="Number of batches for DataLoader benchmark (default: 40).",
    )
    parser.add_argument(
        "--no-dataloader-bench", action="store_true",
        help="Skip the DataLoader comparison benchmark.",
    )
    args = parser.parse_args()

    run_benchmark(
        num_samples=args.num_samples,
        worker_counts=args.workers,
        scaling_sizes=args.scaling_sizes,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        dataloader_batches=args.dataloader_batches,
        run_dataloader_bench=not args.no_dataloader_bench,
    )


if __name__ == "__main__":
    main()
