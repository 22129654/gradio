"""A/B comparison: main vs static-workers branch.

Generates markdown report + charts comparing client latency, server phases,
background traffic, and throughput.

Usage:
    python scripts/benchmark/analyze_static_workers.py \
        --baseline ~/sources/backend-benchmarks/main \
        --test ~/sources/backend-benchmarks/multiprocess-gradio-test \
        --output-dir benchmark_results/static_workers_ab
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

APPS = ["image_to_image", "audio_to_audio", "video_to_video"]
APP_LABELS = {
    "image_to_image": "Image",
    "audio_to_audio": "Audio",
    "video_to_video": "Video",
}

PHASE_COLORS = {
    "queue_wait": "#e74c3c",
    "preprocess": "#9b59b6",
    "fn_call": "#3498db",
    "postprocess": "#f39c12",
    "streaming_diff": "#1abc9c",
}

CLIENT_BREAKDOWN_COLORS = {
    "Upload": "#ff7f0e",
    "Queue Wait": "#9b59b6",
    "Server Processing": "#2ca02c",
    "SSE / Other": "#7f7f7f",
}

BRANCH_COLORS = {"baseline": "#e74c3c", "test": "#3498db"}


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_traces(run_dir: str, app: str) -> tuple[list[dict], list[dict]]:
    """Load traces, splitting upload vs processing."""
    pattern = f"{run_dir}/{app}/*/tier_100/traces.jsonl"
    files = glob.glob(pattern)
    if not files:
        return [], []
    uploads, processing = [], []
    with open(files[0]) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("fn_name") == "gradio_file_upload":
                uploads.append(obj)
            else:
                processing.append(obj)
    return uploads, processing


def load_client_latencies(run_dir: str, app: str) -> list[dict]:
    pattern = f"{run_dir}/{app}/*/tier_100/client_latencies.jsonl"
    files = glob.glob(pattern)
    if not files:
        return []
    with open(files[0]) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_background(run_dir: str, app: str, kind: str) -> list[dict]:
    """Load background_{downloads,uploads,page_loads}.jsonl."""
    pattern = f"{run_dir}/{app}/*/tier_100/background_{kind}.jsonl"
    files = glob.glob(pattern)
    if not files:
        return []
    with open(files[0]) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_summary(run_dir: str, app: str) -> dict | None:
    pattern = f"{run_dir}/{app}/*/summary.json"
    files = glob.glob(pattern)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def load_run_params(run_dir: str) -> dict:
    path = Path(run_dir) / "run_params.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def med(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(values))


def pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_client_metrics(
    traces: list[dict], client_lats: list[dict]
) -> dict:
    successful = [c for c in client_lats if c.get("success")]
    lats = [c["latency_ms"] for c in successful]
    uploads = [c["upload_ms"] for c in successful if c.get("upload_ms") is not None]

    queue_waits = [t["queue_wait_ms"] for t in traces]
    server_totals = [t["total_ms"] for t in traces]

    client_p50 = med(lats)
    upload_p50 = med(uploads)
    queue_p50 = med(queue_waits)
    server_p50 = med(server_totals)
    sse_other = max(0, client_p50 - upload_p50 - queue_p50 - server_p50)

    return {
        "count": len(client_lats),
        "success_count": len(successful),
        "success_rate": len(successful) / len(client_lats) if client_lats else 0,
        "p50": med(lats),
        "p90": pct(lats, 90),
        "p99": pct(lats, 99),
        "upload_p50": upload_p50,
        "queue_wait_p50": queue_p50,
        "server_p50": server_p50,
        "sse_other": sse_other,
        # Server phase breakdown
        "preprocess_p50": med([t["preprocess_ms"] for t in traces]),
        "fn_call_p50": med([t["fn_call_ms"] for t in traces]),
        "postprocess_p50": med([t["postprocess_ms"] for t in traces]),
    }


def compute_background_metrics(run_dir: str, app: str) -> dict:
    result = {}
    for kind in ["downloads", "uploads", "page_loads"]:
        data = load_background(run_dir, app, kind)
        successful = [d for d in data if d.get("success")]
        lats = [d["latency_ms"] for d in successful]
        entry = {
            "count": len(data),
            "success_count": len(successful),
            "success_rate": len(successful) / len(data) if data else 0,
            "p50": med(lats),
            "p90": pct(lats, 90),
            "p99": pct(lats, 99),
        }
        if kind == "page_loads":
            by_endpoint: dict[str, list[float]] = {}
            counts: dict[str, int] = {}
            for d in data:
                ep = d.get("endpoint", "?")
                counts[ep] = counts.get(ep, 0) + 1
                if d.get("success"):
                    by_endpoint.setdefault(ep, []).append(d["latency_ms"])
            entry["by_endpoint"] = {
                ep: {
                    "count": counts[ep],
                    "success_count": len(vals),
                    "p50": med(vals),
                    "p90": pct(vals, 90),
                    "p99": pct(vals, 99),
                }
                for ep, vals in sorted(by_endpoint.items(), key=lambda kv: -med(kv[1]))
            }
        result[kind] = entry
    return result


def compute_throughput(summary: dict | None) -> dict:
    if not summary or not summary.get("tiers"):
        return {"elapsed": 0, "rps": 0, "success_rate": 0}
    tier = summary["tiers"][0]
    elapsed = tier.get("elapsed_seconds", 0)
    total = tier.get("total_requests", 0)
    cs = tier.get("client_summary", {})
    return {
        "elapsed": elapsed,
        "rps": total / elapsed if elapsed > 0 else 0,
        "total_requests": total,
        "success_rate": cs.get("success_rate", 0),
    }


# ── Charts ───────────────────────────────────────────────────────────────────


def plot_client_latency(all_metrics: dict, output_dir: Path, tier):
    """Grouped bar chart: p50/p90/p99 per app, baseline vs test."""
    percentiles = ["p50", "p90", "p99"]
    pct_colors = {"p50": "#2ecc71", "p90": "#f39c12", "p99": "#e74c3c"}

    fig, axes = plt.subplots(1, len(APPS), figsize=(5 * len(APPS), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(APPS):
        ax = axes[idx]
        labels = ["main", "static workers"]
        x = np.arange(len(labels))
        n_pcts = len(percentiles)
        width = 0.8 / n_pcts

        for i, p in enumerate(percentiles):
            vals = [
                all_metrics[app]["baseline"]["client"].get(p, 0),
                all_metrics[app]["test"]["client"].get(p, 0),
            ]
            bars = ax.bar(
                x + i * width, vals, width, label=p if idx == 0 else "",
                color=pct_colors[p], alpha=0.85,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}ms", ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x + width)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(APP_LABELS[app], fontsize=11)
        if idx == 0:
            ax.set_ylabel("Latency (ms)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.suptitle(f"Client Latency Comparison (tier={tier})", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "client_latency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_client_breakdown(all_metrics: dict, output_dir: Path, tier):
    """Stacked bars: upload / queue / server / SSE per branch per app."""
    phases = ["Upload", "Queue Wait", "Server Processing", "SSE / Other"]
    keys = ["upload_p50", "queue_wait_p50", "server_p50", "sse_other"]

    fig, axes = plt.subplots(1, len(APPS), figsize=(5 * len(APPS), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(APPS):
        ax = axes[idx]
        labels = ["main", "static workers"]
        x = np.arange(len(labels))

        bottom = np.zeros(len(labels))
        for phase, key in zip(phases, keys):
            vals = np.array([
                all_metrics[app]["baseline"]["client"].get(key, 0),
                all_metrics[app]["test"]["client"].get(key, 0),
            ])
            ax.bar(
                x, vals, bottom=bottom, label=phase if idx == 0 else "",
                color=CLIENT_BREAKDOWN_COLORS[phase], alpha=0.85, width=0.6,
            )
            bottom += vals

        # Total label on top
        for i, total in enumerate(bottom):
            ax.text(i, total, f"{total:.2f}ms", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(APP_LABELS[app], fontsize=11)
        if idx == 0:
            ax.set_ylabel("Time (ms)")
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.suptitle(f"Latency Breakdown (p50, tier={tier})", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "client_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_background_traffic(all_metrics: dict, output_dir: Path, tier: str):
    """Grouped bars: background download/upload/page_load p50 per branch per app."""
    kinds = ["downloads", "uploads", "page_loads"]
    kind_labels = {"downloads": "Downloads", "uploads": "Uploads", "page_loads": "Page Loads"}

    fig, axes = plt.subplots(1, len(APPS), figsize=(5 * len(APPS), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(APPS):
        ax = axes[idx]
        x = np.arange(len(kinds))
        width = 0.35

        baseline_vals = [all_metrics[app]["baseline"]["background"][k]["p50"] for k in kinds]
        test_vals = [all_metrics[app]["test"]["background"][k]["p50"] for k in kinds]

        bars1 = ax.bar(x - width / 2, baseline_vals, width, label="main" if idx == 0 else "",
                        color=BRANCH_COLORS["baseline"], alpha=0.85)
        bars2 = ax.bar(x + width / 2, test_vals, width, label="static workers" if idx == 0 else "",
                        color=BRANCH_COLORS["test"], alpha=0.85)

        for bar, val in zip(list(bars1) + list(bars2), baseline_vals + test_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}ms", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([kind_labels[k] for k in kinds], fontsize=8)
        ax.set_title(APP_LABELS[app], fontsize=11)
        if idx == 0:
            ax.set_ylabel("p50 Latency (ms)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.suptitle("Background Traffic Latency (p50, tier={tier})", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "background_traffic.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_server_phases(all_metrics: dict, output_dir: Path, tier):
    """Stacked bar: server phase breakdown per branch per app."""
    phases = ["preprocess", "fn_call", "postprocess"]
    keys = ["preprocess_p50", "fn_call_p50", "postprocess_p50"]

    fig, axes = plt.subplots(1, len(APPS), figsize=(5 * len(APPS), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(APPS):
        ax = axes[idx]
        labels = ["main", "static workers"]
        x = np.arange(len(labels))

        bottom = np.zeros(len(labels))
        for phase, key in zip(phases, keys):
            vals = np.array([
                all_metrics[app]["baseline"]["client"].get(key, 0),
                all_metrics[app]["test"]["client"].get(key, 0),
            ])
            ax.bar(
                x, vals, bottom=bottom, label=phase if idx == 0 else "",
                color=PHASE_COLORS[phase], alpha=0.85, width=0.6,
            )
            bottom += vals

        for i, total in enumerate(bottom):
            ax.text(i, total, f"{total:.0f}ms", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(APP_LABELS[app], fontsize=11)
        if idx == 0:
            ax.set_ylabel("Time (ms)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=max(ax.get_ylim()[1] * 1.15, 1))

    fig.suptitle(f"Server Phase Breakdown (p50, tier={tier})", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "server_phases.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Report ───────────────────────────────────────────────────────────────────


def write_report(
    all_metrics: dict,
    baseline_params: dict,
    test_params: dict,
    output_dir: Path,
):
    lines: list[str] = []

    def w(s=""):
        lines.append(s)

    w("# Static Workers A/B Test Results")
    w()
    w("## Configuration")
    w()
    w(f"- **Tier:** {baseline_params.get('tiers', '?')} concurrent users")
    w(f"- **Concurrency limit:** {baseline_params.get('concurrency_limit', '?')}")
    w(f"- **Requests per user:** {baseline_params.get('requests_per_user', '?')}")
    w(f"- **Max threads:** {baseline_params.get('max_threads', '?')}")
    w(f"- **Mode:** {baseline_params.get('mode', '?')} with mixed traffic")
    w(f"- **Baseline:** `main` ({baseline_params.get('commit_sha', '?')[:12]})")
    w(f"- **Test:** `multiprocess-gradio-test` ({test_params.get('commit_sha', '?')[:12]})")
    w()

    # ── Client Latency Table ──
    w("## Client Latency")
    w()
    w("| App | Branch | p50 | p90 | p99 | Improvement (p50) |")
    w("|-----|--------|-----|-----|-----|-------------------|")

    for app in APPS:
        bm = all_metrics[app]["baseline"]["client"]
        tm = all_metrics[app]["test"]["client"]
        factor = bm["p50"] / tm["p50"] if tm["p50"] > 0 else 0

        w(f"| {APP_LABELS[app]} | main | {bm['p50']:.2f}ms | {bm['p90']:.2f}ms | {bm['p99']:.2f}ms | |")
        w(f"| | static workers | {tm['p50']:.2f}ms | {tm['p90']:.2f}ms | {tm['p99']:.2f}ms | **{factor:.2f}x** |")

    w()
    w("![Client Latency](client_latency.png)")
    w()

    # ── Latency Breakdown ──
    w("## Latency Breakdown (p50)")
    w()
    w("| App | Branch | Upload | Queue Wait | Server | SSE/Other |")
    w("|-----|--------|--------|------------|--------|-----------|")

    for app in APPS:
        bm = all_metrics[app]["baseline"]["client"]
        tm = all_metrics[app]["test"]["client"]
        w(f"| {APP_LABELS[app]} | main | {bm['upload_p50']:.2f}ms | {bm['queue_wait_p50']:.2f}ms | {bm['server_p50']:.2f}ms | {bm['sse_other']:.2f}ms |")
        w(f"| | static workers | {tm['upload_p50']:.2f}ms | {tm['queue_wait_p50']:.2f}ms | {tm['server_p50']:.2f}ms | {tm['sse_other']:.2f}ms |")

    w()
    w("![Client Breakdown](client_breakdown.png)")
    w()

    # ── Background Traffic Table ──
    w("## Background Traffic (Mixed Traffic)")
    w()
    w("| App | Traffic Type | Branch | Count | Success Rate | p50 | p99 |")
    w("|-----|-------------|--------|-------|-------------|-----|-----|")

    for app in APPS:
        for kind, kind_label in [("downloads", "Downloads"), ("uploads", "Uploads"), ("page_loads", "Page Loads")]:
            bb = all_metrics[app]["baseline"]["background"][kind]
            tb = all_metrics[app]["test"]["background"][kind]
            w(f"| {APP_LABELS[app]} | {kind_label} | main | {bb['count']} | {bb['success_rate']:.1%} | {bb['p50']:.2f}ms | {bb['p99']:.2f}ms |")
            w(f"| | | static workers | {tb['count']} | {tb['success_rate']:.1%} | {tb['p50']:.2f}ms | {tb['p99']:.2f}ms |")

    w()
    w("![Background Traffic](background_traffic.png)")
    w()

    # ── Page Load Breakdown by Endpoint ──
    w("## Page Load Latency by Endpoint")
    w()
    w("Top endpoints by p50 latency. Helps isolate whether slow page loads are coming")
    w("from the HTML/config routes (hit `gradio_main`) or static assets (hit `static_workers`).")
    w()
    w("| App | Branch | Endpoint | Count | p50 | p90 | p99 |")
    w("|-----|--------|----------|-------|-----|-----|-----|")

    for app in APPS:
        for branch_label, branch_key in [("main", "baseline"), ("static workers", "test")]:
            pl = all_metrics[app][branch_key]["background"]["page_loads"]
            by_ep = pl.get("by_endpoint") or {}
            if not by_ep:
                continue
            for ep, s in list(by_ep.items())[:8]:
                ep_disp = ep if len(ep) <= 40 else ep[:37] + "..."
                w(
                    f"| {APP_LABELS[app]} | {branch_label} | `{ep_disp}` | "
                    f"{s['count']} | {s['p50']:.2f}ms | {s['p90']:.2f}ms | {s['p99']:.2f}ms |"
                )
    w()

    # ── Server Phases ──
    w("## Server Phase Breakdown (p50)")
    w()
    w("| App | Branch | Preprocess | fn_call | Postprocess | Total |")
    w("|-----|--------|------------|---------|-------------|-------|")

    for app in APPS:
        bm = all_metrics[app]["baseline"]["client"]
        tm = all_metrics[app]["test"]["client"]
        bt = bm["preprocess_p50"] + bm["fn_call_p50"] + bm["postprocess_p50"]
        tt = tm["preprocess_p50"] + tm["fn_call_p50"] + tm["postprocess_p50"]
        w(f"| {APP_LABELS[app]} | main | {bm['preprocess_p50']:.0f}ms | {bm['fn_call_p50']:.0f}ms | {bm['postprocess_p50']:.0f}ms | {bt:.0f}ms |")
        w(f"| | static workers | {tm['preprocess_p50']:.0f}ms | {tm['fn_call_p50']:.0f}ms | {tm['postprocess_p50']:.0f}ms | {tt:.0f}ms |")

    w()
    w("![Server Phases](server_phases.png)")
    w()

    # ── Throughput ──
    w("## Throughput & Reliability")
    w()
    w("| App | Branch | Elapsed | Requests | RPS | Success Rate |")
    w("|-----|--------|---------|----------|-----|-------------|")

    for app in APPS:
        bt = all_metrics[app]["baseline"]["throughput"]
        tt = all_metrics[app]["test"]["throughput"]
        w(f"| {APP_LABELS[app]} | main | {bt['elapsed']:.0f}ms | {bt.get('total_requests', '?')} | {bt['rps']:.1f} | {bt['success_rate']:.1%} |")
        w(f"| | static workers | {tt['elapsed']:.0f}ms | {tt.get('total_requests', '?')} | {tt['rps']:.1f} | {tt['success_rate']:.1%} |")

    w()

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare main vs static-workers benchmark results"
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline (main) results")
    parser.add_argument("--test", required=True, help="Path to test (static workers) results")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/static_workers_ab",
        help="Output directory",
    )
    args = parser.parse_args()

    baseline_dir = str(Path(args.baseline).expanduser())
    test_dir = str(Path(args.test).expanduser())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_params = load_run_params(baseline_dir)
    test_params = load_run_params(test_dir)

    all_metrics: dict = {}

    for app in APPS:
        print(f"Loading {app}...")

        _, b_traces = load_traces(baseline_dir, app)
        b_clients = load_client_latencies(baseline_dir, app)
        b_summary = load_summary(baseline_dir, app)

        _, t_traces = load_traces(test_dir, app)
        t_clients = load_client_latencies(test_dir, app)
        t_summary = load_summary(test_dir, app)

        all_metrics[app] = {
            "baseline": {
                "client": compute_client_metrics(b_traces, b_clients),
                "background": compute_background_metrics(baseline_dir, app),
                "throughput": compute_throughput(b_summary),
            },
            "test": {
                "client": compute_client_metrics(t_traces, t_clients),
                "background": compute_background_metrics(test_dir, app),
                "throughput": compute_throughput(t_summary),
            },
        }

    print("\nGenerating charts...")
    plot_client_latency(all_metrics, output_dir, 100)
    print("  -> client_latency.png")
    plot_client_breakdown(all_metrics, output_dir, 100)
    print("  -> client_breakdown.png")
    plot_background_traffic(all_metrics, output_dir, 100)
    print("  -> background_traffic.png")
    plot_server_phases(all_metrics, output_dir, 100)
    print("  -> server_phases.png")

    print("Writing report...")
    report_path = write_report(all_metrics, baseline_params, test_params, output_dir)
    print(f"  -> {report_path}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
