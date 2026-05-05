"""Three-way comparison of benchmark results.

Generates markdown report + charts comparing multiple branch/config combinations.

Usage:
    python scripts/benchmark/analyze_three_way.py \
        --branches main=/path/to/main \
        --branches "nginx (no redis)=/path/to/no-redis" \
        --branches "nginx + redis=/path/to/redis" \
        --output-dir benchmark_results/three_way
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

BRANCH_COLORS = [
    "#e74c3c",  # red
    "#3498db",  # blue
    "#2ecc71",  # green
    "#f39c12",  # orange
    "#9b59b6",  # purple
]

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


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_traces(run_dir: str, app: str) -> tuple[list[dict], list[dict]]:
    pattern = f"{run_dir}/{app}/*/tier_*/traces.jsonl"
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
    pattern = f"{run_dir}/{app}/*/tier_*/client_latencies.jsonl"
    files = glob.glob(pattern)
    if not files:
        return []
    with open(files[0]) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_background(run_dir: str, app: str, kind: str) -> list[dict]:
    pattern = f"{run_dir}/{app}/*/tier_*/background_{kind}.jsonl"
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


def compute_client_metrics(traces: list[dict], client_lats: list[dict]) -> dict:
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
        result[kind] = {
            "count": len(data),
            "success_count": len(successful),
            "success_rate": len(successful) / len(data) if data else 0,
            "p50": med(lats),
            "p90": pct(lats, 90),
            "p99": pct(lats, 99),
        }
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


def plot_client_latency(all_metrics: dict, branch_names: list[str], output_dir: Path):
    percentiles = ["p50", "p90", "p99"]
    pct_colors = {"p50": "#2ecc71", "p90": "#f39c12", "p99": "#e74c3c"}

    apps = [a for a in APPS if a in all_metrics]
    fig, axes = plt.subplots(1, len(apps), figsize=(5 * len(apps), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(apps):
        ax = axes[idx]
        x = np.arange(len(branch_names))
        n_pcts = len(percentiles)
        width = 0.8 / n_pcts

        for i, p in enumerate(percentiles):
            vals = [
                all_metrics[app][name]["client"].get(p, 0)
                for name in branch_names
            ]
            bars = ax.bar(
                x + i * width, vals, width, label=p if idx == 0 else "",
                color=pct_colors[p], alpha=0.85,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.0f}", ha="center", va="bottom", fontsize=6,
                    )

        ax.set_xticks(x + width)
        ax.set_xticklabels(branch_names, fontsize=7, rotation=15, ha="right")
        ax.set_title(APP_LABELS.get(app, app), fontsize=11)
        if idx == 0:
            ax.set_ylabel("Latency (ms)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.suptitle("Client Latency Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "client_latency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_client_breakdown(all_metrics: dict, branch_names: list[str], output_dir: Path):
    phases = ["Upload", "Queue Wait", "Server Processing", "SSE / Other"]
    keys = ["upload_p50", "queue_wait_p50", "server_p50", "sse_other"]

    apps = [a for a in APPS if a in all_metrics]
    fig, axes = plt.subplots(1, len(apps), figsize=(5 * len(apps), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(apps):
        ax = axes[idx]
        x = np.arange(len(branch_names))

        bottom = np.zeros(len(branch_names))
        for phase, key in zip(phases, keys):
            vals = np.array([
                all_metrics[app][name]["client"].get(key, 0)
                for name in branch_names
            ])
            ax.bar(
                x, vals, bottom=bottom, label=phase if idx == 0 else "",
                color=CLIENT_BREAKDOWN_COLORS[phase], alpha=0.85, width=0.6,
            )
            bottom += vals

        for i, total in enumerate(bottom):
            ax.text(i, total, f"{total:.0f}ms", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(branch_names, fontsize=7, rotation=15, ha="right")
        ax.set_title(APP_LABELS.get(app, app), fontsize=11)
        if idx == 0:
            ax.set_ylabel("Time (ms)")
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.suptitle("Latency Breakdown (p50)", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "client_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_background_traffic(all_metrics: dict, branch_names: list[str], output_dir: Path):
    kinds = ["downloads", "uploads", "page_loads"]
    kind_labels = {"downloads": "Downloads", "uploads": "Uploads", "page_loads": "Page Loads"}

    apps = [a for a in APPS if a in all_metrics]
    fig, axes = plt.subplots(1, len(apps), figsize=(5 * len(apps), 6), squeeze=False)
    axes = axes[0]

    n_branches = len(branch_names)
    width = 0.8 / n_branches

    for idx, app in enumerate(apps):
        ax = axes[idx]
        x = np.arange(len(kinds))

        for bi, name in enumerate(branch_names):
            vals = [all_metrics[app][name]["background"][k]["p50"] for k in kinds]
            bars = ax.bar(
                x + bi * width - (n_branches - 1) * width / 2, vals, width,
                label=name if idx == 0 else "",
                color=BRANCH_COLORS[bi % len(BRANCH_COLORS)], alpha=0.85,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.0f}", ha="center", va="bottom", fontsize=6,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([kind_labels[k] for k in kinds], fontsize=8)
        ax.set_title(APP_LABELS.get(app, app), fontsize=11)
        if idx == 0:
            ax.set_ylabel("p50 Latency (ms)")
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.suptitle("Background Traffic Latency (p50)", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "background_traffic.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_server_phases(all_metrics: dict, branch_names: list[str], output_dir: Path):
    phases = ["preprocess", "fn_call", "postprocess"]
    keys = ["preprocess_p50", "fn_call_p50", "postprocess_p50"]

    apps = [a for a in APPS if a in all_metrics]
    fig, axes = plt.subplots(1, len(apps), figsize=(5 * len(apps), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(apps):
        ax = axes[idx]
        x = np.arange(len(branch_names))

        bottom = np.zeros(len(branch_names))
        for phase, key in zip(phases, keys):
            vals = np.array([
                all_metrics[app][name]["client"].get(key, 0)
                for name in branch_names
            ])
            ax.bar(
                x, vals, bottom=bottom, label=phase if idx == 0 else "",
                color=PHASE_COLORS[phase], alpha=0.85, width=0.6,
            )
            bottom += vals

        for i, total in enumerate(bottom):
            ax.text(i, total, f"{total:.0f}ms", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(branch_names, fontsize=7, rotation=15, ha="right")
        ax.set_title(APP_LABELS.get(app, app), fontsize=11)
        if idx == 0:
            ax.set_ylabel("Time (ms)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=max(ax.get_ylim()[1] * 1.15, 1))

    fig.suptitle("Server Phase Breakdown (p50)", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "server_phases.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_throughput(all_metrics: dict, branch_names: list[str], output_dir: Path):
    apps = [a for a in APPS if a in all_metrics]
    fig, axes = plt.subplots(1, len(apps), figsize=(5 * len(apps), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(apps):
        ax = axes[idx]
        x = np.arange(len(branch_names))
        vals = [all_metrics[app][name]["throughput"]["rps"] for name in branch_names]
        bars = ax.bar(
            x, vals, color=[BRANCH_COLORS[i % len(BRANCH_COLORS)] for i in range(len(branch_names))],
            alpha=0.85, width=0.6,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(branch_names, fontsize=7, rotation=15, ha="right")
        ax.set_title(APP_LABELS.get(app, app), fontsize=11)
        if idx == 0:
            ax.set_ylabel("Requests/sec")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.suptitle("Throughput (requests/sec)", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "throughput.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Report ───────────────────────────────────────────────────────────────────


def write_report(
    all_metrics: dict,
    branch_names: list[str],
    branch_params: dict[str, dict],
    output_dir: Path,
):
    lines: list[str] = []

    def w(s=""):
        lines.append(s)

    apps = [a for a in APPS if a in all_metrics]
    first_params = branch_params[branch_names[0]]

    w("# Benchmark Comparison Results")
    w()
    w("## Configuration")
    w()
    w(f"- **Tier:** {first_params.get('tiers', '?')} concurrent users")
    w(f"- **Concurrency limit:** {first_params.get('concurrency_limit', '?')}")
    w(f"- **Requests per user:** {first_params.get('requests_per_user', '?')}")
    w(f"- **Mode:** {first_params.get('mode', '?')}")
    w()
    w("| Branch | Commit | Workers | Nginx | Redis |")
    w("|--------|--------|---------|-------|-------|")
    for name in branch_names:
        p = branch_params[name]
        w(f"| {name} | `{p.get('commit_sha', '?')[:12]}` | {p.get('num_workers', 1)} | {p.get('nginx', False)} | {p.get('redis', False)} |")
    w()

    # ── Client Latency ──
    w("## Client Latency")
    w()
    header = "| App | Branch | p50 | p90 | p99 |"
    sep = "|-----|--------|-----|-----|-----|"
    w(header)
    w(sep)
    for app in apps:
        for name in branch_names:
            m = all_metrics[app][name]["client"]
            w(f"| {APP_LABELS.get(app, app)} | {name} | {m['p50']:.0f}ms | {m['p90']:.0f}ms | {m['p99']:.0f}ms |")
    w()
    w("![Client Latency](client_latency.png)")
    w()

    # ── Latency Breakdown ──
    w("## Latency Breakdown (p50)")
    w()
    w("| App | Branch | Upload | Queue Wait | Server | SSE/Other |")
    w("|-----|--------|--------|------------|--------|-----------|")
    for app in apps:
        for name in branch_names:
            m = all_metrics[app][name]["client"]
            w(f"| {APP_LABELS.get(app, app)} | {name} | {m['upload_p50']:.0f}ms | {m['queue_wait_p50']:.0f}ms | {m['server_p50']:.0f}ms | {m['sse_other']:.0f}ms |")
    w()
    w("![Client Breakdown](client_breakdown.png)")
    w()

    # ── Background Traffic ──
    w("## Background Traffic (p50)")
    w()
    w("| App | Traffic | Branch | Count | Success | p50 | p90 |")
    w("|-----|---------|--------|-------|---------|-----|-----|")
    for app in apps:
        for kind, kind_label in [("downloads", "Downloads"), ("uploads", "Uploads"), ("page_loads", "Page Loads")]:
            for name in branch_names:
                b = all_metrics[app][name]["background"][kind]
                w(f"| {APP_LABELS.get(app, app)} | {kind_label} | {name} | {b['count']} | {b['success_rate']:.0%} | {b['p50']:.0f}ms | {b['p90']:.0f}ms |")
    w()
    w("![Background Traffic](background_traffic.png)")
    w()

    # ── Server Phases ──
    w("## Server Phase Breakdown (p50)")
    w()
    w("| App | Branch | Preprocess | fn_call | Postprocess | Total |")
    w("|-----|--------|------------|---------|-------------|-------|")
    for app in apps:
        for name in branch_names:
            m = all_metrics[app][name]["client"]
            total = m["preprocess_p50"] + m["fn_call_p50"] + m["postprocess_p50"]
            w(f"| {APP_LABELS.get(app, app)} | {name} | {m['preprocess_p50']:.0f}ms | {m['fn_call_p50']:.0f}ms | {m['postprocess_p50']:.0f}ms | {total:.0f}ms |")
    w()
    w("![Server Phases](server_phases.png)")
    w()

    # ── Throughput ──
    w("## Throughput")
    w()
    w("| App | Branch | Elapsed | Requests | RPS | Success Rate |")
    w("|-----|--------|---------|----------|-----|-------------|")
    for app in apps:
        for name in branch_names:
            t = all_metrics[app][name]["throughput"]
            w(f"| {APP_LABELS.get(app, app)} | {name} | {t['elapsed']:.0f}s | {t.get('total_requests', '?')} | {t['rps']:.1f} | {t['success_rate']:.0%} |")
    w()
    w("![Throughput](throughput.png)")
    w()

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results across multiple branches/configs"
    )
    parser.add_argument(
        "--branches",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Branch label and results path (e.g. 'main=/path/to/results'). Repeat for each branch.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/three_way",
        help="Output directory",
    )
    args = parser.parse_args()

    branches: list[tuple[str, str]] = []
    for spec in args.branches:
        if "=" not in spec:
            print(f"ERROR: --branches must be LABEL=PATH, got: {spec}")
            return
        label, path = spec.split("=", 1)
        branches.append((label.strip(), str(Path(path.strip()).expanduser())))

    branch_names = [name for name, _ in branches]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    branch_params = {}
    all_metrics: dict = {}

    for name, run_dir in branches:
        branch_params[name] = load_run_params(run_dir)

    for app in APPS:
        app_data = {}
        has_data = False
        for name, run_dir in branches:
            _, traces = load_traces(run_dir, app)
            clients = load_client_latencies(run_dir, app)
            summary = load_summary(run_dir, app)

            if clients:
                has_data = True

            app_data[name] = {
                "client": compute_client_metrics(traces, clients),
                "background": compute_background_metrics(run_dir, app),
                "throughput": compute_throughput(summary),
            }
        if has_data:
            all_metrics[app] = app_data
            print(f"Loaded {app}")
        else:
            print(f"Skipping {app} (no data)")

    print("\nGenerating charts...")
    plot_client_latency(all_metrics, branch_names, output_dir)
    print("  -> client_latency.png")
    plot_client_breakdown(all_metrics, branch_names, output_dir)
    print("  -> client_breakdown.png")
    plot_background_traffic(all_metrics, branch_names, output_dir)
    print("  -> background_traffic.png")
    plot_server_phases(all_metrics, branch_names, output_dir)
    print("  -> server_phases.png")
    plot_throughput(all_metrics, branch_names, output_dir)
    print("  -> throughput.png")

    print("Writing report...")
    report_path = write_report(all_metrics, branch_names, branch_params, output_dir)
    print(f"  -> {report_path}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
