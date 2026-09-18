#!/usr/bin/env python3
"""
bench_utils.py — Shared utilities for the VLM benchmark suite.

Provides:
  - load_real_dataset()   : Streaming HuggingFace dataset → VQASample list.
  - apply_bright_style()  : Set matplotlib rcParams for the bright aesthetic.
  - save_fig()            : Uniform figure-save wrapper.
  - BRIGHT_PALETTE        : Curated vibrant color list.

All benchmark scripts import from this module so aesthetic and loading
logic stays DRY.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Bootstrap: ensure project root is importable
_BENCH_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCH_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.schema import VQASample

logger = logging.getLogger(__name__)

# Color palette — bright, vibrant, visually distinct
BRIGHT_PALETTE: list[str] = [
    "#4C9BE8",  # vivid sky blue
    "#F4845F",  # warm coral
    "#56C490",  # mint green
    "#A97CF7",  # soft violet
    "#F7C948",  # golden yellow
    "#E05C8A",  # hot pink
    "#3BC9C0",  # teal
    "#FF9A3C",  # tangerine
]

# Static vs Dynamic semantic colours
COLOR_STATIC  = "#F4845F"   # coral / warm
COLOR_DYNAMIC = "#4C9BE8"   # sky blue / cool
COLOR_IMAGE   = "#F7C948"   # gold (image component)
COLOR_TEXT    = "#A97CF7"   # violet (text component)
COLOR_SAVINGS = "#56C490"   # mint green (savings / positive delta)
COLOR_SPEEDUP = "#E05C8A"   # pink (speedup line overlay)

# Dataset specifications — tried in order until one succeeds
_DATASET_SPECS: list[dict[str, Any]] = [
    # merve/vqav2-small: small public VQAv2 subset, real JPEG images
    {
        "dataset_id": "merve/vqav2-small",
        "config_name": None,
        "split": "validation",
        "col_image": "image",
        "col_question": "question",
        "col_answer": "multiple_choice_answer",
        "is_caption_list": False,
        "dataset_name": "VQAv2-small",
    },
    # HuggingFaceM4/NoCaps: public, no auth, real JPEG images + captions
    {
        "dataset_id": "HuggingFaceM4/NoCaps",
        "config_name": None,
        "split": "validation",
        "col_image": "image",
        "col_question": "file_name",
        "col_answer": "annotations_captions",
        "is_caption_list": True,
        "dataset_name": "NoCaps",
    },
    # nlphuji/flickr30k: Fallback Flickr30k captions
    {
        "dataset_id": "nlphuji/flickr30k",
        "config_name": None,
        "split": "test",
        "col_image": "image",
        "col_question": "filename",
        "col_answer": "caption",
        "is_caption_list": True,
        "dataset_name": "Flickr30k",
    },
]


def load_real_dataset(
    num_samples: int,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> list[VQASample]:
    """Load real VQA/captioning samples from a public HuggingFace dataset.

    Tries dataset sources in priority order, falling back to the next if
    the current one fails.  Uses HuggingFace streaming so only the first
    ``num_samples`` rows are ever downloaded.

    Parameters
    ----------
    num_samples : int
        Maximum number of samples to collect.
    seed : int
        Random seed used to shuffle the collected samples in-memory.
    cache_dir : str | Path | None
        HuggingFace cache directory.  ``None`` uses the HF default.

    Returns
    -------
    list[VQASample]
        List of validated VQASample objects from real data.
    """
    from datasets import load_dataset as hf_load  # noqa: PLC0415

    samples: list[VQASample] = []
    last_exc: Exception | None = None

    for spec in _DATASET_SPECS:
        dataset_id = spec["dataset_id"]
        config_name = spec["config_name"]
        split = spec["split"]

        print(f"  → Trying dataset: {dataset_id!r} (split={split!r}) ...")
        try:
            load_kwargs: dict[str, Any] = {
                "streaming": True,
                "trust_remote_code": True,
            }
            if config_name:
                load_kwargs["name"] = config_name
            if cache_dir:
                load_kwargs["cache_dir"] = str(cache_dir)

            ds = hf_load(dataset_id, split=split, **load_kwargs)

            fetched = 0
            skipped = 0
            t0 = time.perf_counter()

            for row_idx, row in enumerate(ds):
                if fetched >= num_samples:
                    break

                try:
                    sample = _row_to_vqa_sample(row, spec, row_idx)
                    samples.append(sample)
                    fetched += 1

                    if fetched % 200 == 0:
                        elapsed = time.perf_counter() - t0
                        rate = fetched / elapsed if elapsed > 0 else 0
                        print(
                            f"    Fetched {fetched}/{num_samples} samples "
                            f"({elapsed:.1f}s, {rate:.0f} rows/s) ..."
                        )
                except Exception as row_exc:
                    skipped += 1
                    if skipped <= 5:
                        logger.debug("Skipping row %d: %s", row_idx, row_exc)
                    continue

            elapsed = time.perf_counter() - t0
            print(
                f"  ✓ Loaded {len(samples)} real samples from {dataset_id!r} "
                f"({skipped} skipped, {elapsed:.1f}s total)"
            )

            if samples:
                # Shuffle in-memory for reproducibility across benchmark runs
                rng = np.random.RandomState(seed)
                indices = rng.permutation(len(samples)).tolist()
                samples = [samples[i] for i in indices]
                return samples

        except Exception as exc:
            print(f"    ✗ Failed ({type(exc).__name__}): {exc}")
            last_exc = exc
            samples = []
            continue

    raise RuntimeError(
        f"All dataset sources failed to load real data. "
        f"Last error: {last_exc}"
    ) from last_exc


def _row_to_vqa_sample(
    row: dict[str, Any],
    spec: dict[str, Any],
    row_idx: int,
) -> VQASample:
    """Convert a single HF dataset row to a VQASample using the spec mapping."""
    from PIL import Image  # noqa: PLC0415

    # ── Image ──────────────────────────────────────────────────────────────
    image_value = row.get(spec["col_image"])
    if image_value is None:
        raise ValueError(f"Missing image column '{spec['col_image']}'")

    if isinstance(image_value, Image.Image):
        img = image_value.convert("RGB")
    elif isinstance(image_value, bytes):
        img = Image.open(io.BytesIO(image_value)).convert("RGB")
    elif isinstance(image_value, dict):
        raw_bytes = image_value.get("bytes")
        raw_path  = image_value.get("path")
        if raw_bytes:
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        elif raw_path:
            img = Image.open(raw_path).convert("RGB")
        else:
            raise ValueError("Image dict has neither 'bytes' nor 'path'")
    else:
        raise ValueError(f"Unsupported image type: {type(image_value).__name__}")

    width, height = img.size
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    image_bytes = buf.getvalue()

    # ── Question ────────────────────────────────────────────────────────────
    question_raw = row.get(spec["col_question"], "")
    if isinstance(question_raw, list):
        question = str(question_raw[0]) if question_raw else ""
    else:
        question = str(question_raw or "")
    question = question.strip() or f"Describe this image (sample {row_idx})"

    # ── Answer ──────────────────────────────────────────────────────────────
    answer_raw = row.get(spec["col_answer"], "")
    if spec.get("is_caption_list") and isinstance(answer_raw, list):
        answer = str(answer_raw[0]) if answer_raw else ""
    else:
        answer = str(answer_raw or "")
    answer = answer.strip() or "No caption available."

    # ── ID ──────────────────────────────────────────────────────────────────
    sample_id = str(
        row.get("image_id", row.get("id", f"real_{row_idx:07d}"))
    )

    sample = VQASample(
        sample_id=sample_id,
        image_bytes=image_bytes,
        image_width=width,
        image_height=height,
        question=question,
        answer=answer,
        metadata={"source_row": row_idx, "source_dataset": spec["dataset_name"]},
        dataset_name=spec["dataset_name"],
    )
    sample.validate()
    return sample


# Matplotlib bright aesthetic

def apply_bright_style() -> None:
    """Set global matplotlib rcParams for the bright, spacious benchmark style.

    Characteristics:
    - White / off-white backgrounds (no dark mode)
    - Clean, readable dashed grid lines at low alpha
    - Large, bold fonts for readability at presentation size
    - Tight layout with room reserved for outside-area legends
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")  # non-interactive, safe for headless scripts
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update({
        # ── Canvas ──────────────────────────────────────────────────────────
        "figure.facecolor":     "white",
        "axes.facecolor":       "#F8F9FA",
        "axes.edgecolor":       "#CCCCCC",
        "axes.linewidth":       1.3,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        # ── Labels & titles ──────────────────────────────────────────────────
        "axes.labelcolor":      "#1A252F",
        "axes.labelsize":       14,
        "axes.labelweight":     "bold",
        "axes.titlesize":       17,
        "axes.titleweight":     "bold",
        "axes.titlepad":        16,
        # ── Ticks ────────────────────────────────────────────────────────────
        "xtick.color":          "#555555",
        "ytick.color":          "#555555",
        "xtick.labelsize":      12,
        "ytick.labelsize":      12,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.pad":      6,
        "ytick.major.pad":      6,
        # ── Grid ─────────────────────────────────────────────────────────────
        "axes.grid":            True,
        "axes.axisbelow":       True,
        "grid.color":           "#DDDDDD",
        "grid.linestyle":       "--",
        "grid.linewidth":       0.85,
        "grid.alpha":           0.7,
        # ── Legend ───────────────────────────────────────────────────────────
        "legend.fontsize":      12,
        "legend.framealpha":    0.95,
        "legend.edgecolor":     "#CCCCCC",
        "legend.facecolor":     "white",
        "legend.borderpad":     0.8,
        "legend.handlelength":  2.0,
        # ── Text ─────────────────────────────────────────────────────────────
        "text.color":           "#1A252F",
        "font.family":          "sans-serif",
        "font.size":            12,
        # ── Lines & markers ──────────────────────────────────────────────────
        "lines.linewidth":      2.4,
        "lines.markersize":     9,
        # ── Patches ──────────────────────────────────────────────────────────
        "patch.linewidth":      0.6,
        "patch.edgecolor":      "white",
        # ── Figure ───────────────────────────────────────────────────────────
        "figure.dpi":           100,
        "figure.autolayout":    False,
        "savefig.dpi":          150,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
        "savefig.pad_inches":   0.25,
    })


def save_fig(fig: Any, path: str | Path, *, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str | Path
        Output file path (directory is created if it does not exist).
    dpi : int
        Resolution in dots-per-inch.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        pad_inches=0.25,
    )
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


def add_bar_labels(
    ax: Any,
    bars: Any,
    fmt: str = "{:.1f}",
    color: str = "#1A252F",
    fontsize: int = 11,
    padding_frac: float = 0.015,
) -> None:
    """Place bold value labels directly above each bar in a bar chart.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes containing the bars.
    bars : matplotlib.container.BarContainer
        Return value of ``ax.bar()``.
    fmt : str
        Python format string for the label text.
    color : str
        Label text colour.
    fontsize : int
        Label font size.
    padding_frac : float
        Fraction of the tallest bar's height to use as vertical padding.
    """
    all_heights = [b.get_height() for b in bars if b.get_height() > 0]
    max_h = max(all_heights, default=1.0) or 1.0

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max_h * padding_frac,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold", color=color,
            )
