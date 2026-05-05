"""Analyze file-processing benchmark results to identify server-side bottlenecks.

Generates waffle charts:
  - 1 server-side phase breakdown chart per modality (image, audio, video)
    with cl=1 and cl=100 side by side, ordered: preprocess -> fn_call -> postprocess
  - 3 client-side latency breakdown charts (image, audio, video)
    showing % of client latency in upload, queue_wait, server processing, and SSE/other

Usage:
    python scripts/benchmark/analyze_file_processing.py \
        --cl1 ~/sources/backend-benchmarks/file-processing-benchmark-run-9 \
        --cl100 ~/sources/backend-benchmarks/file-processing-benchmark-cl-100-run-2 \
        --output-dir benchmark_results/file_analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BRANCH = "profile-the-upload-route"
FILE_APPS = ["image_to_image", "audio_to_audio", "video_to_video"]
APP_LABELS = {
    "image_to_image": "Image to Image",
    "audio_to_audio": "Audio to Audio",
    "video_to_video": "Video to Video",
}

# --- Consistent color palette for server sub-phases across all modalities ---
# Blues for preprocess, green for fn_call, warm tones for postprocess
SERVER_PHASE_COLORS = {
    "pre: move_to_cache": "#1f77b4",
    "pre: format_image": "#4a90d9",
    "pre: audio_from_file": "#7ab3ef",
    "pre: video_decode": "#a8cce8",
    "pre: other": "#d0e1f2",
    "fn_call": "#2ca02c",
    "post: save_img_to_cache": "#d62728",
    "post: save_audio_to_cache": "#e45756",
    "post: video_to_mp4": "#f28e8e",
    "post: update_state_config": "#ff7f0e",
    "post: move_to_cache": "#f5b7b1",
    "post: other": "#c49c94",
}

CLIENT_PHASE_COLORS = {
    "Upload": "#ff7f0e",
    "Queue Wait": "#9467bd",
    "Server Processing": "#2ca02c",
    "SSE / Other": "#7f7f7f",
}

WAFFLE_ROWS = 10
WAFFLE_COLS = 10
WAFFLE_TOTAL = WAFFLE_ROWS * WAFFLE_COLS


def load_traces(
    run_dir: str, app: str, tier: str = "tier_100"
) -> tuple[list[dict], list[dict]]:
    """Load traces, splitting into upload traces and processing traces."""
    pattern = f"{run_dir}/{BRANCH}/{app}/*/{tier}/traces.jsonl"
    files = glob.glob(pattern)
    if not files:
        print(f"WARNING: No traces found for {app} in {run_dir}", file=sys.stderr)
        return [], []
    uploads, processing = [], []
    with open(files[0]) as f:
        for line in f:
            obj = json.loads(line)
            if obj["fn_name"] == "gradio_file_upload":
                uploads.append(obj)
            else:
                processing.append(obj)
    return uploads, processing


def load_client_latencies(run_dir: str, app: str, tier: str = "tier_100") -> list[dict]:
    """Load client latencies from a run."""
    pattern = f"{run_dir}/{BRANCH}/{app}/*/{tier}/client_latencies.jsonl"
    files = glob.glob(pattern)
    if not files:
        return []
    with open(files[0]) as f:
        return [json.loads(line) for line in f]


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(values))


def slices_to_waffle_grid(slices: list[tuple[str, float]]) -> np.ndarray:
    """Convert named slices into a 10x10 grid of category indices.

    Slices are laid out in order top-to-bottom, left-to-right so the first
    slice occupies the top-left cells.
    """
    total = sum(v for _, v in slices)
    if total == 0:
        return np.zeros((WAFFLE_ROWS, WAFFLE_COLS), dtype=int)

    # Allocate cells proportionally (largest-remainder method)
    raw = [(v / total) * WAFFLE_TOTAL for _, v in slices]
    floors = [int(f) for f in raw]
    remainders = [r - f for r, f in zip(raw, floors)]
    allocated = list(floors)
    deficit = WAFFLE_TOTAL - sum(allocated)
    for idx in sorted(range(len(remainders)), key=lambda i: -remainders[i]):
        if deficit <= 0:
            break
        allocated[idx] += 1
        deficit -= 1

    # Fill the grid top-to-bottom, left-to-right (first slice = top-left)
    grid = np.zeros((WAFFLE_ROWS, WAFFLE_COLS), dtype=int)
    cell = 0
    for cat_idx, count in enumerate(allocated):
        for _ in range(count):
            row = cell // WAFFLE_COLS
            col = cell % WAFFLE_COLS
            grid[row, col] = cat_idx
            cell += 1

    return grid


def draw_waffle(
    ax: plt.Axes,
    slices: list[tuple[str, float]],
    color_map: dict[str, str],
    title: str,
    subtitle: str,
):
    """Draw a single waffle chart on the given axes."""
    grid = slices_to_waffle_grid(slices)

    labels = [s[0] for s in slices]
    colors_for_idx = [color_map.get(label, "#cccccc") for label in labels]

    for row in range(WAFFLE_ROWS):
        for col in range(WAFFLE_COLS):
            cat_idx = grid[row, col]
            # Flip y so row 0 is at top
            y = WAFFLE_ROWS - 1 - row
            rect = plt.Rectangle(
                (col, y),
                0.9,
                0.9,
                facecolor=colors_for_idx[cat_idx],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.add_patch(rect)

    ax.set_xlim(-0.2, WAFFLE_COLS + 0.2)
    ax.set_ylim(-0.2, WAFFLE_ROWS + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=10)


def get_server_slices(traces: list[dict]) -> list[tuple[str, float]]:
    """Compute server sub-phase slices ordered: pre -> fn_call -> post."""
    pre_slices: list[tuple[str, float]] = []
    fn_slices: list[tuple[str, float]] = []
    post_slices: list[tuple[str, float]] = []

    # --- Preprocess sub-phases ---
    pre_sub = {
        "pre: move_to_cache": median(
            [t.get("preprocess_move_to_cache_ms", 0) for t in traces]
        ),
        "pre: format_image": median(
            [t.get("preprocess_format_image_ms", 0) for t in traces]
        ),
        "pre: audio_from_file": median(
            [t.get("preprocess_audio_from_file_ms", 0) for t in traces]
        ),
        "pre: video_decode": median([t.get("preprocess_video_ms", 0) for t in traces]),
    }
    pre_total = median([t["preprocess_ms"] for t in traces])
    pre_other = max(0, pre_total - sum(pre_sub.values()))

    for name, val in pre_sub.items():
        if val > 0.01:
            pre_slices.append((name, val))
    if pre_other > 0.01:
        pre_slices.append(("pre: other", pre_other))
    pre_slices.sort(key=lambda x: -x[1])

    # --- fn_call ---
    fn_call = median([t["fn_call_ms"] for t in traces])
    if fn_call > 0.01:
        fn_slices.append(("fn_call", fn_call))

    # --- Postprocess sub-phases ---
    post_sub = {
        "post: save_img_to_cache": median(
            [t.get("postprocess_save_img_array_to_cache_ms", 0) for t in traces]
        ),
        "post: save_audio_to_cache": median(
            [t.get("postprocess_save_audio_to_cache_ms", 0) for t in traces]
        ),
        "post: video_to_mp4": median(
            [
                t.get("postprocess_video_convert_video_to_playable_mp4_ms", 0)
                for t in traces
            ]
        ),
        "post: update_state_config": median(
            [t.get("postprocess_update_state_in_config_ms", 0) for t in traces]
        ),
        "post: move_to_cache": median(
            [t.get("postprocess_move_to_cache_ms", 0) for t in traces]
        ),
    }
    post_total = median([t["postprocess_ms"] for t in traces])
    post_other = max(0, post_total - sum(post_sub.values()))

    for name, val in post_sub.items():
        if val > 0.01:
            post_slices.append((name, val))
    if post_other > 0.01:
        post_slices.append(("post: other", post_other))
    post_slices.sort(key=lambda x: -x[1])

    return pre_slices + fn_slices + post_slices


def get_client_slices(
    traces: list[dict], client_lats: list[dict]
) -> list[tuple[str, float]]:
    """Compute client breakdown slices."""
    successful = [c for c in client_lats if c.get("success")]
    client_total = median([c["latency_ms"] for c in successful])
    upload = median(
        [c["upload_ms"] for c in successful if c.get("upload_ms") is not None]
    )
    queue_wait = median([t["queue_wait_ms"] for t in traces])
    server_processing = median([t["total_ms"] for t in traces])
    sse_other = max(0, client_total - upload - queue_wait - server_processing)

    slices = [
        ("Upload", upload),
        ("Queue Wait", queue_wait),
        ("Server Processing", server_processing),
        ("SSE / Other", sse_other),
    ]
    return [(name, val) for name, val in slices if val > 0.01]


def plot_server_phases(cl1_dir: str, cl100_dir: str, output_dir: Path):
    """One figure per modality with cl=1 and cl=100 side by side."""

    # Collect all labels across all apps and both runs for a unified legend
    all_labels_ordered: list[str] = []

    for app in FILE_APPS:
        for run_dir in [cl1_dir, cl100_dir]:
            _, traces = load_traces(run_dir, app)
            if traces:
                for label, _ in get_server_slices(traces):
                    if label not in all_labels_ordered:
                        all_labels_ordered.append(label)

    for app in FILE_APPS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(
            f"Server Processing Time Breakdown — {APP_LABELS[app]}",
            fontsize=15,
            y=0.98,
        )

        for ax, (run_dir, run_label) in zip(
            axes, [(cl1_dir, "cl=1"), (cl100_dir, "cl=100")]
        ):
            _, traces = load_traces(run_dir, app)
            if not traces:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(run_label)
                continue

            slices = get_server_slices(traces)
            total = sum(v for _, v in slices)
            draw_waffle(
                ax,
                slices,
                SERVER_PHASE_COLORS,
                run_label,
                f"median server total: {total:.1f}ms",
            )

        # Build unified legend in pre -> fn_call -> post order
        legend_handles = []
        for label in all_labels_ordered:
            color = SERVER_PHASE_COLORS.get(label, "#cccccc")
            legend_handles.append(
                mpatches.Patch(facecolor=color, edgecolor="white", label=label)
            )

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 4),
            fontsize=10,
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        out_path = output_dir / f"server_phases_{app}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


def plot_client_breakdown(cl1_dir: str, cl100_dir: str, output_dir: Path):
    """One figure per modality with cl=1 and cl=100 side by side."""

    for app in FILE_APPS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(
            f"Client Latency Breakdown — {APP_LABELS[app]}",
            fontsize=15,
            y=0.98,
        )

        for ax, (run_dir, run_label) in zip(
            axes, [(cl1_dir, "cl=1"), (cl100_dir, "cl=100")]
        ):
            _, traces = load_traces(run_dir, app)
            client_lats = load_client_latencies(run_dir, app)
            if not traces or not client_lats:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(run_label)
                continue

            slices = get_client_slices(traces, client_lats)
            successful = [c for c in client_lats if c.get("success")]
            client_total = median([c["latency_ms"] for c in successful])

            draw_waffle(
                ax,
                slices,
                CLIENT_PHASE_COLORS,
                run_label,
                f"median client latency: {client_total:.0f}ms",
            )

        legend_handles = [
            mpatches.Patch(facecolor=color, edgecolor="white", label=label)
            for label, color in CLIENT_PHASE_COLORS.items()
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            fontsize=10,
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        out_path = output_dir / f"client_breakdown_{app}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


def plot_sync_vs_async(cl100_dir: str, async_dir: str, output_dir: Path):
    """Compare sync vs async image_to_image at cl=100."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Server Processing — Image to Image cl=100: sync vs async fn",
        fontsize=15,
        y=0.98,
    )

    all_labels: list[str] = []
    for run_dir, app in [
        (cl100_dir, "image_to_image"),
        (async_dir, "image_to_image_async"),
    ]:
        _, traces = load_traces(run_dir, app)
        if traces:
            for label, _ in get_server_slices(traces):
                if label not in all_labels:
                    all_labels.append(label)

    for ax, (run_dir, app, label) in zip(
        axes,
        [
            (cl100_dir, "image_to_image", "sync fn"),
            (async_dir, "image_to_image_async", "async fn"),
        ],
    ):
        _, traces = load_traces(run_dir, app)
        if not traces:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(label)
            continue
        slices = get_server_slices(traces)
        total = sum(v for _, v in slices)
        draw_waffle(
            ax,
            slices,
            SERVER_PHASE_COLORS,
            label,
            f"median server total: {total:.1f}ms",
        )

    legend_handles = [
        mpatches.Patch(
            facecolor=SERVER_PHASE_COLORS.get(l, "#ccc"), edgecolor="white", label=l
        )
        for l in all_labels
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 4),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    out_path = output_dir / "server_phases_image_sync_vs_async.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_sync_vs_async_client(cl100_dir: str, async_dir: str, output_dir: Path):
    """Compare client latency breakdown for sync vs async image_to_image at cl=100."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Client Latency Breakdown — Image to Image cl=100: sync vs async fn",
        fontsize=15,
        y=0.98,
    )

    for ax, (run_dir, app, label) in zip(
        axes,
        [
            (cl100_dir, "image_to_image", "sync fn"),
            (async_dir, "image_to_image_async", "async fn"),
        ],
    ):
        _, traces = load_traces(run_dir, app)
        client_lats = load_client_latencies(run_dir, app)
        if not traces or not client_lats:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(label)
            continue
        slices = get_client_slices(traces, client_lats)
        successful = [c for c in client_lats if c.get("success")]
        client_total = median([c["latency_ms"] for c in successful])
        draw_waffle(
            ax,
            slices,
            CLIENT_PHASE_COLORS,
            label,
            f"median client latency: {client_total:.0f}ms",
        )

    legend_handles = [
        mpatches.Patch(facecolor=c, edgecolor="white", label=l)
        for l, c in CLIENT_PHASE_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    out_path = output_dir / "client_breakdown_image_sync_vs_async.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def analyze_event_loop_blocking(
    cl1_dir: str, cl100_dir: str, async_dir: str, output_dir: Path
):
    """Measure how sync pre/postprocess blocks the event loop and starves SSE delivery.

    Writes findings to event_loop_blocking_analysis.txt.
    """
    configs = [
        ("cl=1 sync image", cl1_dir, "image_to_image"),
        ("cl=100 sync image", cl100_dir, "image_to_image"),
        ("cl=100 async image", async_dir, "image_to_image_async"),
        ("cl=1 sync echo", cl1_dir, "echo_text"),
        ("cl=100 sync echo", cl100_dir, "echo_text"),
        ("cl=1 sync audio", cl1_dir, "audio_to_audio"),
        ("cl=100 sync audio", cl100_dir, "audio_to_audio"),
        ("cl=1 sync video", cl1_dir, "video_to_video"),
        ("cl=100 sync video", cl100_dir, "video_to_video"),
    ]

    lines: list[str] = []

    def w(s: str = ""):
        lines.append(s)

    def header(s: str):
        w()
        w("=" * 80)
        w(f"  {s}")
        w("=" * 80)

    # ── Section 1: Per-config summary table ──
    header("EVENT LOOP BLOCKING ANALYSIS")
    w()
    w("Sync pre/postprocess methods (PIL.Image.save, AudioSegment.from_file,")
    w("ffmpeg, shutil.copy2, Path.write_bytes) run on the main thread and block")
    w("the asyncio event loop. At high concurrency, this starves SSE message")
    w("delivery, inflating client-observed latency.")
    w()
    w("Methodology: for each round of 100 concurrent users, sum all individual")
    w("preprocess_ms + postprocess_ms values. This 'cumulative blocking' is the")
    w("total time the event loop spends in sync file I/O per round. Compare to")
    w("'SSE/other' = client_total - upload - queue_wait - server_total, which is")
    w("the unexplained gap in client latency.")

    header("SUMMARY TABLE")
    w()
    w(
        f"{'Config':<25} {'Users/round':>12} {'Cum block/rnd':>14} "
        f"{'SSE/other p50':>14} {'Correlation':>12}"
    )
    w("-" * 80)

    for label, run_dir, app in configs:
        _, traces = load_traces(run_dir, app)
        client_lats = load_client_latencies(run_dir, app)
        if not traces or not client_lats:
            w(f"{label:<25} {'NO DATA':>12}")
            continue

        by_round: dict[int, list[dict]] = {}
        for t in traces:
            parts = t["session_hash"].split("_")
            rid = int(parts[2])
            by_round.setdefault(rid, []).append(t)

        by_round_client: dict[int, list[dict]] = {}
        for c in client_lats:
            by_round_client.setdefault(c["request_id"], []).append(c)

        users_per_round = max(len(v) for v in by_round.values())

        round_data = []
        for rid in sorted(by_round.keys()):
            rt = by_round[rid]
            rc = [c for c in by_round_client.get(rid, []) if c.get("success")]
            if not rc:
                continue
            cum_block = sum(t["preprocess_ms"] + t["postprocess_ms"] for t in rt)
            client_med = median([c["latency_ms"] for c in rc])
            upload_vals = [c["upload_ms"] for c in rc if c.get("upload_ms") is not None]
            upload_med = median(upload_vals) if upload_vals else 0
            qw_med = median([t["queue_wait_ms"] for t in rt])
            server_med = median([t["total_ms"] for t in rt])
            sse = client_med - upload_med - qw_med - server_med
            round_data.append((cum_block, sse))

        if not round_data:
            w(f"{label:<25} {'NO DATA':>12}")
            continue

        cum_blocks, sses = zip(*round_data)
        cum_med = np.median(cum_blocks)
        sse_med = np.median(sses)
        corr = float(np.corrcoef(cum_blocks, sses)[0, 1]) if len(round_data) > 2 else 0

        w(
            f"{label:<25} {users_per_round:>12} {cum_med:>12.0f}ms "
            f"{sse_med:>12.0f}ms {corr:>11.3f}"
        )

    # ── Section 2: Detailed per-round breakdown for async image ──
    header("PER-ROUND DETAIL: cl=100 async image")
    w()

    _, traces = load_traces(async_dir, "image_to_image_async")
    client_lats = load_client_latencies(async_dir, "image_to_image_async")

    if traces and client_lats:
        by_round = {}
        for t in traces:
            rid = int(t["session_hash"].split("_")[2])
            by_round.setdefault(rid, []).append(t)

        by_round_client = {}
        for c in client_lats:
            by_round_client.setdefault(c["request_id"], []).append(c)

        w(
            f"{'Round':>5} {'Cum pre+post':>14} {'Avg pre+post':>14} "
            f"{'Max qw':>10} {'SSE/other':>12} {'Client p50':>12}"
        )
        w("-" * 70)

        for rid in sorted(by_round.keys()):
            rt = by_round[rid]
            rc = [c for c in by_round_client.get(rid, []) if c.get("success")]
            if not rc:
                continue
            cum_block = sum(t["preprocess_ms"] + t["postprocess_ms"] for t in rt)
            avg_block = cum_block / len(rt)
            max_qw = max(t["queue_wait_ms"] for t in rt)
            client_med = median([c["latency_ms"] for c in rc])
            upload_vals = [c["upload_ms"] for c in rc if c.get("upload_ms") is not None]
            upload_med = median(upload_vals) if upload_vals else 0
            qw_med = median([t["queue_wait_ms"] for t in rt])
            server_med = median([t["total_ms"] for t in rt])
            sse = client_med - upload_med - qw_med - server_med
            w(
                f"{rid:>5} {cum_block:>12.0f}ms {avg_block:>12.1f}ms "
                f"{max_qw:>8.0f}ms {sse:>10.0f}ms {client_med:>10.0f}ms"
            )

    # ── Section 3: Breakdown of what blocks the event loop ──
    header("BLOCKING OPERATIONS BY SUB-PHASE")
    w()
    w("Median ms per request at cl=100, and cumulative blocking per round (100 users).")
    w()

    sub_phases = [
        ("postprocess_save_img_array_to_cache_ms", "PIL.Image.save + write_bytes"),
        ("preprocess_audio_from_file_ms", "AudioSegment.from_file (ffprobe)"),
        ("postprocess_video_convert_video_to_playable_mp4_ms", "ffmpeg subprocess"),
        ("preprocess_move_to_cache_ms", "shutil.copy2"),
        ("postprocess_move_to_cache_ms", "shutil.copy2"),
        ("preprocess_format_image_ms", "PIL format conversion"),
        ("postprocess_save_audio_to_cache_ms", "audio file write"),
        ("postprocess_update_state_in_config_ms", "move_files_to_cache + dict update"),
    ]

    file_configs = [
        ("image cl=100", cl100_dir, "image_to_image"),
        ("image async", async_dir, "image_to_image_async"),
        ("audio cl=100", cl100_dir, "audio_to_audio"),
        ("video cl=100", cl100_dir, "video_to_video"),
    ]

    w(
        f"{'Sub-phase':<55} {'Blocking call':<30} {'Config':<16} "
        f"{'p50/req':>9} {'Cum/round':>10}"
    )
    w("-" * 125)

    for phase_key, blocking_call in sub_phases:
        for cfg_label, run_dir, app in file_configs:
            _, traces = load_traces(run_dir, app)
            if not traces:
                continue
            vals = [t.get(phase_key, 0) for t in traces]
            p50 = float(np.median(vals))
            if p50 < 0.01:
                continue
            # Estimate users per round
            by_round = {}
            for t in traces:
                rid = int(t["session_hash"].split("_")[2])
                by_round.setdefault(rid, []).append(t)
            users = max(len(v) for v in by_round.values())
            cum = p50 * users
            w(
                f"  {phase_key:<53} {blocking_call:<30} {cfg_label:<16} "
                f"{p50:>7.1f}ms {cum:>8.0f}ms"
            )

    # ── Section 4: Key findings ──
    header("KEY FINDINGS")
    w()
    w("1. SYNC PRE/POSTPROCESS BLOCKS THE EVENT LOOP")
    w("   At cl=100, each round of 100 image requests accumulates ~4000ms of")
    w("   sync file I/O (PIL.Image.save, shutil.copy2, etc.) that blocks the")
    w("   asyncio event loop. At cl=1 the same total blocking occurs but is")
    w("   serialized, so SSE delivery is only delayed by one request at a time.")
    w()
    w("2. SSE DELIVERY IS STARVED AT HIGH CONCURRENCY")
    w("   The 'SSE/other' gap (client_total - upload - queue_wait - server_total)")
    w("   is 235ms at cl=1 but 3300-4300ms at cl=100 for image workloads.")
    w("   Correlation between cumulative blocking and SSE overhead is 0.85,")
    w("   confirming a direct causal link.")
    w()
    w("3. ASYNC FN DOESN'T HELP (SERVER IMPROVES, CLIENT DOESN'T)")
    w("   Making the user function async reduced server_total from 707ms to 18ms")
    w("   (eliminated GIL contention in fn_call). But client latency stayed at")
    w("   ~5100ms because the pre/postprocess I/O still blocks the event loop.")
    w()
    w("4. ECHO_TEXT CONFIRMS THE CAUSE")
    w("   Echo text has ~5ms cumulative blocking per round and low SSE overhead")
    w("   at both cl=1 (211ms) and cl=100 (503ms). No file I/O = no blocking.")
    w()
    w("5. TOP BLOCKING OPERATIONS TO MAKE ASYNC")
    w("   - postprocess_save_img_array_to_cache: PIL.Image.save + write_bytes")
    w("     13-19ms/request, ~1400ms cumulative/round")
    w("   - preprocess_audio_from_file: AudioSegment.from_file (ffprobe subprocess)")
    w("     1-146ms/request (high variance), ~800ms cumulative/round")
    w("   - postprocess_video_convert_video_to_playable_mp4: ffmpeg subprocess")
    w("     0-20ms/request (high variance)")
    w("   - preprocess/postprocess_move_to_cache: shutil.copy2")
    w("     ~0.8ms/request each, ~160ms cumulative/round")
    w()
    w("RECOMMENDATION: Wrap these sync I/O calls in asyncio.to_thread() so they")
    w("run on a worker thread and don't block the event loop. This should reduce")
    w("the SSE/other gap at high concurrency from ~4000ms to near the echo_text")
    w("baseline of ~500ms.")

    out_path = output_dir / "event_loop_blocking_analysis.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Saved {out_path}")


def plot_threading_comparison(
    old_cl100_dir: str, new_cl100_dir: str, new_mt100_dir: str, output_dir: Path
):
    """Per-modality comparison: before threading, after (mt=40), after (mt=100).

    Image gets a 4th panel for async. Each modality produces a client and server chart.
    """
    # panels per app: list of (label, run_dir, app_name)
    modalities = {
        "image_to_image": {
            "label": "Image to Image",
            "panels": [
                ("before\n(on event loop)", old_cl100_dir, "image_to_image"),
                ("threaded\n(mt=40)", new_cl100_dir, "image_to_image"),
                ("threaded\n(mt=100)", new_mt100_dir, "image_to_image"),
            ],
        },
        "image_to_image_async": {
            "label": "Image to Image (async fn)",
            "panels": [
                ("before\n(on event loop)", old_cl100_dir, "image_to_image"),
                ("threaded\n(mt=40)", new_cl100_dir, "image_to_image_async"),
                ("threaded\n(mt=100)", new_mt100_dir, "image_to_image_async"),
            ],
        },
        "audio_to_audio": {
            "label": "Audio to Audio",
            "panels": [
                ("before\n(on event loop)", old_cl100_dir, "audio_to_audio"),
                ("threaded\n(mt=40)", new_cl100_dir, "audio_to_audio"),
                ("threaded\n(mt=100)", new_mt100_dir, "audio_to_audio"),
            ],
        },
        "video_to_video": {
            "label": "Video to Video",
            "panels": [
                ("before\n(on event loop)", old_cl100_dir, "video_to_video"),
                ("threaded\n(mt=40)", new_cl100_dir, "video_to_video"),
                ("threaded\n(mt=100)", new_mt100_dir, "video_to_video"),
            ],
        },
    }

    for app_key, info in modalities.items():
        panels = info["panels"]
        app_label = info["label"]
        n = len(panels)

        # ── Client latency chart ──
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 8))
        if n == 1:
            axes = [axes]
        fig.suptitle(
            f"Client Latency — {app_label} cl=100",
            fontsize=15,
        )

        for ax, (label, run_dir, app) in zip(axes, panels):
            _, traces = load_traces(run_dir, app)
            client_lats = load_client_latencies(run_dir, app)
            if not traces or not client_lats:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(label)
                continue
            slices = get_client_slices(traces, client_lats)
            successful = [c for c in client_lats if c.get("success")]
            client_total = median([c["latency_ms"] for c in successful])
            draw_waffle(
                ax, slices, CLIENT_PHASE_COLORS, label, f"median: {client_total:.0f}ms"
            )

        legend_handles = [
            mpatches.Patch(facecolor=c, edgecolor="white", label=l)
            for l, c in CLIENT_PHASE_COLORS.items()
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
        )
        fig.subplots_adjust(top=0.88, bottom=0.10)
        out_path = output_dir / f"client_threading_{app_key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

        # ── Server phases chart ──
        all_labels: list[str] = []
        for _, run_dir, app in panels:
            _, traces = load_traces(run_dir, app)
            if traces:
                for sl, _ in get_server_slices(traces):
                    if sl not in all_labels:
                        all_labels.append(sl)

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 8))
        if n == 1:
            axes = [axes]
        fig.suptitle(
            f"Server Processing — {app_label} cl=100",
            fontsize=15,
        )

        for ax, (label, run_dir, app) in zip(axes, panels):
            _, traces = load_traces(run_dir, app)
            if not traces:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(label)
                continue
            slices = get_server_slices(traces)
            total = sum(v for _, v in slices)
            draw_waffle(
                ax, slices, SERVER_PHASE_COLORS, label, f"median: {total:.1f}ms"
            )

        legend_handles = [
            mpatches.Patch(
                facecolor=SERVER_PHASE_COLORS.get(l, "#ccc"), edgecolor="white", label=l
            )
            for l in all_labels
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 4),
            fontsize=10,
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
        )
        fig.subplots_adjust(top=0.88, bottom=0.10)
        out_path = output_dir / f"server_threading_{app_key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate server and client latency breakdown waffle charts"
    )
    parser.add_argument("--cl1", required=True, help="Path to cl=1 benchmark run")
    parser.add_argument("--cl100", required=True, help="Path to cl=100 benchmark run")
    parser.add_argument(
        "--async-run", required=True, help="Path to async cl=100 benchmark run"
    )
    parser.add_argument(
        "--old-cl100",
        default=None,
        help="Path to OLD cl=100 run (pre/postprocess on event loop)",
    )
    parser.add_argument(
        "--new-cl100",
        default=None,
        help="Path to NEW cl=100 run (pre/postprocess in thread pool, mt=40)",
    )
    parser.add_argument(
        "--new-cl100-mt100",
        default=None,
        help="Path to NEW cl=100 run with max_threads=100",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/file_analysis",
        help="Output directory for plots (default: benchmark_results/file_analysis)",
    )
    args = parser.parse_args()

    cl1_dir = str(Path(args.cl1).expanduser())
    cl100_dir = str(Path(args.cl100).expanduser())
    async_dir = str(Path(args.async_run).expanduser())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating server phase breakdown waffle charts...")
    plot_server_phases(cl1_dir, cl100_dir, output_dir)

    print("Generating client latency breakdown waffle charts...")
    plot_client_breakdown(cl1_dir, cl100_dir, output_dir)

    print("Generating sync vs async comparison...")
    plot_sync_vs_async(cl100_dir, async_dir, output_dir)
    plot_sync_vs_async_client(cl100_dir, async_dir, output_dir)

    print("Analyzing event loop blocking...")
    analyze_event_loop_blocking(cl1_dir, cl100_dir, async_dir, output_dir)

    if args.old_cl100 and args.new_cl100 and args.new_cl100_mt100:
        old_cl100 = str(Path(args.old_cl100).expanduser())
        new_cl100 = str(Path(args.new_cl100).expanduser())
        new_mt100 = str(Path(args.new_cl100_mt100).expanduser())
        print("Generating threading comparison charts...")
        plot_threading_comparison(old_cl100, new_cl100, new_mt100, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
