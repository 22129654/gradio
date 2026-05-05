"""Three-way worker comparison: main vs N workers (no nginx) vs N workers (nginx).

Generates charts comparing client latency, server phases, background traffic,
and throughput across three configurations, for a given tier.

Usage:
    python scripts/benchmark/analyze_workers.py \
        --data-dir ~/sources/backend-benchmarks/worker-comparisons-2 \
        --tier 100 \
        --output-dir benchmark_results/worker_comparison

    # Compare both tiers side by side
    python scripts/benchmark/analyze_workers.py \
        --data-dir ~/sources/backend-benchmarks/worker-comparisons-2 \
        --tier 100 200 \
        --output-dir benchmark_results/worker_comparison
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

BRANCHES = [
    ("main-profiling-fix", "main"),
    ("multiprocess-gradio-test-no-nginx", "workers (no nginx)"),
    ("multiprocess-gradio-test-nginx", "workers (nginx)"),
]

BRANCH_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]

PHASE_COLORS = {
    "preprocess": "#9b59b6",
    "fn_call": "#3498db",
    "postprocess": "#f39c12",
}

CLIENT_BREAKDOWN_COLORS = {
    "Upload": "#ff7f0e",
    "Queue Wait": "#9b59b6",
    "Server Processing": "#2ca02c",
    "SSE / Other": "#7f7f7f",
}


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_traces(run_dir: str, app: str) -> list[dict]:
    pattern = f"{run_dir}/{app}/*/tier_*/traces.jsonl"
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    with open(files[-1]) as f:
        return [
            json.loads(line) for line in f if line.strip()
            and json.loads(line).get("fn_name") != "gradio_file_upload"
        ]


def load_client_latencies(run_dir: str, app: str) -> list[dict]:
    pattern = f"{run_dir}/{app}/*/tier_*/client_latencies.jsonl"
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    with open(files[-1]) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_background(run_dir: str, app: str, kind: str) -> list[dict]:
    pattern = f"{run_dir}/{app}/*/tier_*/background_{kind}.jsonl"
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    with open(files[-1]) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_summary(run_dir: str, app: str) -> dict | None:
    pattern = f"{run_dir}/{app}/*/summary.json"
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def load_run_params(run_dir: str) -> dict:
    path = Path(run_dir) / "run_params.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def med(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def pct(values: list[float], p: int) -> float:
    return float(np.percentile(values, p)) if values else 0.0


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
        return {"elapsed": 0, "rps": 0, "success_rate": 0, "total_requests": 0}
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


def load_tier_metrics(data_dir: str, tier: int) -> dict:
    tier_dir = f"{data_dir}/tier_{tier}"
    branch_labels = [label for _, label in BRANCHES]
    metrics = {}

    for app in APPS:
        app_data = {}
        has_data = False
        for (branch_dir, label) in BRANCHES:
            run_dir = f"{tier_dir}/{branch_dir}"
            traces = load_traces(run_dir, app)
            clients = load_client_latencies(run_dir, app)
            summary = load_summary(run_dir, app)
            if clients:
                has_data = True
            app_data[label] = {
                "client": compute_client_metrics(traces, clients),
                "background": compute_background_metrics(run_dir, app),
                "throughput": compute_throughput(summary),
            }
        if has_data:
            metrics[app] = app_data
    return metrics


# ── Charts ───────────────────────────────────────────────────────────────────


def plot_client_latency(all_metrics: dict, branch_labels: list[str], tier: int, output_dir: Path):
    percentiles = ["p50", "p90", "p99"]
    pct_colors = {"p50": "#2ecc71", "p90": "#f39c12", "p99": "#e74c3c"}

    apps = [a for a in APPS if a in all_metrics]
    fig, axes = plt.subplots(1, len(apps), figsize=(5 * len(apps), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(apps):
        ax = axes[idx]
        x = np.arange(len(branch_labels))
        n_pcts = len(percentiles)
        width = 0.8 / n_pcts

        for i, p in enumerate(percentiles):
            vals = [all_metrics[app][name]["client"].get(p, 0) for name in branch_labels]
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
        ax.set_xticklabels(branch_labels, fontsize=7, rotation=15, ha="right")
        ax.set_title(APP_LABELS.get(app, app), fontsize=11)
        if idx == 0:
            ax.set_ylabel("Latency (ms)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.suptitle(f"Client Latency — tier {tier}", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / f"client_latency_tier{tier}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_client_breakdown(all_metrics: dict, branch_labels: list[str], tier: int, output_dir: Path):
    phases = ["Upload", "Queue Wait", "Server Processing", "SSE / Other"]
    keys = ["upload_p50", "queue_wait_p50", "server_p50", "sse_other"]

    apps = [a for a in APPS if a in all_metrics]
    fig, axes = plt.subplots(1, len(apps), figsize=(5 * len(apps), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(apps):
        ax = axes[idx]
        x = np.arange(len(branch_labels))

        bottom = np.zeros(len(branch_labels))
        for phase, key in zip(phases, keys):
            vals = np.array([
                all_metrics[app][name]["client"].get(key, 0) for name in branch_labels
            ])
            ax.bar(
                x, vals, bottom=bottom, label=phase if idx == 0 else "",
                color=CLIENT_BREAKDOWN_COLORS[phase], alpha=0.85, width=0.6,
            )
            bottom += vals

        for i, total in enumerate(bottom):
            ax.text(i, total, f"{total:.0f}ms", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(branch_labels, fontsize=7, rotation=15, ha="right")
        ax.set_title(APP_LABELS.get(app, app), fontsize=11)
        if idx == 0:
            ax.set_ylabel("Time (ms)")
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.suptitle(f"Latency Breakdown (p50) — tier {tier}", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / f"client_breakdown_tier{tier}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_background_traffic(all_metrics: dict, branch_labels: list[str], tier: int, output_dir: Path):
    kinds = ["downloads", "uploads", "page_loads"]
    kind_labels = {"downloads": "Downloads", "uploads": "Uploads", "page_loads": "Page Loads"}

    apps = [a for a in APPS if a in all_metrics]
    fig, axes = plt.subplots(1, len(apps), figsize=(5 * len(apps), 6), squeeze=False)
    axes = axes[0]

    n_branches = len(branch_labels)
    width = 0.8 / n_branches

    for idx, app in enumerate(apps):
        ax = axes[idx]
        x = np.arange(len(kinds))

        for bi, name in enumerate(branch_labels):
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

    fig.suptitle(f"Background Traffic Latency (p50) — tier {tier}", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / f"background_traffic_tier{tier}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_server_phases(all_metrics: dict, branch_labels: list[str], tier: int, output_dir: Path):
    phases = ["preprocess", "fn_call", "postprocess"]
    keys = ["preprocess_p50", "fn_call_p50", "postprocess_p50"]

    apps = [a for a in APPS if a in all_metrics]
    fig, axes = plt.subplots(1, len(apps), figsize=(5 * len(apps), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(apps):
        ax = axes[idx]
        x = np.arange(len(branch_labels))

        bottom = np.zeros(len(branch_labels))
        for phase, key in zip(phases, keys):
            vals = np.array([
                all_metrics[app][name]["client"].get(key, 0) for name in branch_labels
            ])
            ax.bar(
                x, vals, bottom=bottom, label=phase if idx == 0 else "",
                color=PHASE_COLORS[phase], alpha=0.85, width=0.6,
            )
            bottom += vals

        for i, total in enumerate(bottom):
            ax.text(i, total, f"{total:.0f}ms", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(branch_labels, fontsize=7, rotation=15, ha="right")
        ax.set_title(APP_LABELS.get(app, app), fontsize=11)
        if idx == 0:
            ax.set_ylabel("Time (ms)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=max(ax.get_ylim()[1] * 1.15, 1))

    fig.suptitle(f"Server Phase Breakdown (p50) — tier {tier}", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / f"server_phases_tier{tier}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_throughput(all_metrics: dict, branch_labels: list[str], tier: int, output_dir: Path):
    apps = [a for a in APPS if a in all_metrics]
    fig, axes = plt.subplots(1, len(apps), figsize=(5 * len(apps), 6), squeeze=False)
    axes = axes[0]

    for idx, app in enumerate(apps):
        ax = axes[idx]
        x = np.arange(len(branch_labels))
        vals = [all_metrics[app][name]["throughput"]["rps"] for name in branch_labels]
        bars = ax.bar(
            x, vals, color=BRANCH_COLORS[:len(branch_labels)], alpha=0.85, width=0.6,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(branch_labels, fontsize=7, rotation=15, ha="right")
        ax.set_title(APP_LABELS.get(app, app), fontsize=11)
        if idx == 0:
            ax.set_ylabel("Requests/sec")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.suptitle(f"Throughput — tier {tier}", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / f"throughput_tier{tier}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Report ───────────────────────────────────────────────────────────────────


def write_report(
    tier_data: dict[int, dict],
    branch_labels: list[str],
    tier_params: dict[int, dict[str, dict]],
    output_dir: Path,
):
    lines: list[str] = []

    def w(s=""):
        lines.append(s)

    w("# Worker Comparison Results")
    w()

    for tier, all_metrics in sorted(tier_data.items()):
        apps = [a for a in APPS if a in all_metrics]
        params = tier_params[tier]

        w(f"## Tier {tier}")
        w()
        w("| Config | Workers | Nginx | Concurrency Limit |")
        w("|--------|---------|-------|-------------------|")
        for label in branch_labels:
            p = params.get(label, {})
            w(f"| {label} | {p.get('num_workers', 1)} | {p.get('nginx', False)} | {p.get('concurrency_limit', '?')} |")
        w()

        w("### Client Latency")
        w()
        w("| App | Config | p50 | p90 | p99 | Success |")
        w("|-----|--------|-----|-----|-----|---------|")
        for app in apps:
            for name in branch_labels:
                m = all_metrics[app][name]["client"]
                w(f"| {APP_LABELS.get(app, app)} | {name} | {m['p50']:.0f}ms | {m['p90']:.0f}ms | {m['p99']:.0f}ms | {m['success_rate']:.0%} |")
        w()
        w(f"![Client Latency](client_latency_tier{tier}.png)")
        w()

        w("### Latency Breakdown (p50)")
        w()
        w("| App | Config | Upload | Queue | Server | SSE/Other |")
        w("|-----|--------|--------|-------|--------|-----------|")
        for app in apps:
            for name in branch_labels:
                m = all_metrics[app][name]["client"]
                w(f"| {APP_LABELS.get(app, app)} | {name} | {m['upload_p50']:.0f}ms | {m['queue_wait_p50']:.0f}ms | {m['server_p50']:.0f}ms | {m['sse_other']:.0f}ms |")
        w()
        w(f"![Client Breakdown](client_breakdown_tier{tier}.png)")
        w()

        w("### Background Traffic (p50)")
        w()
        w("| App | Traffic | Config | Count | p50 | p90 |")
        w("|-----|---------|--------|-------|-----|-----|")
        for app in apps:
            for kind, kind_label in [("downloads", "Downloads"), ("uploads", "Uploads"), ("page_loads", "Page Loads")]:
                for name in branch_labels:
                    b = all_metrics[app][name]["background"][kind]
                    w(f"| {APP_LABELS.get(app, app)} | {kind_label} | {name} | {b['count']} | {b['p50']:.0f}ms | {b['p90']:.0f}ms |")
        w()
        w(f"![Background Traffic](background_traffic_tier{tier}.png)")
        w()

        w("### Server Phases (p50)")
        w()
        w("| App | Config | Preprocess | fn_call | Postprocess | Total |")
        w("|-----|--------|------------|---------|-------------|-------|")
        for app in apps:
            for name in branch_labels:
                m = all_metrics[app][name]["client"]
                total = m["preprocess_p50"] + m["fn_call_p50"] + m["postprocess_p50"]
                w(f"| {APP_LABELS.get(app, app)} | {name} | {m['preprocess_p50']:.0f}ms | {m['fn_call_p50']:.0f}ms | {m['postprocess_p50']:.0f}ms | {total:.0f}ms |")
        w()
        w(f"![Server Phases](server_phases_tier{tier}.png)")
        w()

        w("### Throughput")
        w()
        w("| App | Config | Elapsed | Requests | RPS | Success |")
        w("|-----|--------|---------|----------|-----|---------|")
        for app in apps:
            for name in branch_labels:
                t = all_metrics[app][name]["throughput"]
                w(f"| {APP_LABELS.get(app, app)} | {name} | {t['elapsed']:.0f}s | {t['total_requests']} | {t['rps']:.1f} | {t['success_rate']:.0%} |")
        w()
        w(f"![Throughput](throughput_tier{tier}.png)")
        w()
        w("---")
        w()

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare main vs workers (no nginx) vs workers (nginx)"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root directory containing tier_N subdirectories",
    )
    parser.add_argument(
        "--tier", nargs="+", type=int, required=True,
        help="Tier(s) to analyze (e.g. 100 200)",
    )
    parser.add_argument(
        "--output-dir", default="benchmark_results/worker_comparison",
        help="Output directory",
    )
    args = parser.parse_args()

    data_dir = str(Path(args.data_dir).expanduser())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    branch_labels = [label for _, label in BRANCHES]
    tier_data = {}
    tier_params = {}

    for tier in args.tier:
        print(f"\nLoading tier {tier}...")
        metrics = load_tier_metrics(data_dir, tier)
        tier_data[tier] = metrics

        params = {}
        for branch_dir, label in BRANCHES:
            params[label] = load_run_params(f"{data_dir}/tier_{tier}/{branch_dir}")
        tier_params[tier] = params

        print(f"  Apps: {list(metrics.keys())}")
        print("  Generating charts...")

        plot_client_latency(metrics, branch_labels, tier, output_dir)
        print(f"    -> client_latency_tier{tier}.png")
        plot_client_breakdown(metrics, branch_labels, tier, output_dir)
        print(f"    -> client_breakdown_tier{tier}.png")
        plot_background_traffic(metrics, branch_labels, tier, output_dir)
        print(f"    -> background_traffic_tier{tier}.png")
        plot_server_phases(metrics, branch_labels, tier, output_dir)
        print(f"    -> server_phases_tier{tier}.png")
        plot_throughput(metrics, branch_labels, tier, output_dir)
        print(f"    -> throughput_tier{tier}.png")

    print("\nWriting report...")
    report_path = write_report(tier_data, branch_labels, tier_params, output_dir)
    print(f"  -> {report_path}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
