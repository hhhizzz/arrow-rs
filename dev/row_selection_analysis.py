#!/usr/bin/env python3
"""
Run the row_selection_state benchmark, export Criterion results to CSV,
and draw line charts comparing the available Criterion groups across
length/type variations.
"""

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

SUFFIX_RE = re.compile(
    r"(?P<suite>[^-]+)-(?P<scenario>[^-]+)-(?P<variant>.+?)-"
    r"L(?P<selector_len>\d+)-avg(?P<avg_selector_len>[0-9.]+)-sel(?P<select_ratio>\d+)"
)


def run_bench(skip: bool, extra_args: List[str]) -> None:
    if skip:
        return
    cmd = ["cargo", "bench", "--bench", "row_selection_state", *extra_args]
    subprocess.run(cmd, check=True)


def load_estimates(case_dir: Path) -> Dict[str, Dict[str, float]]:
    for sub in ("new", "final", "base", ""):
        base = case_dir / sub if sub else case_dir
        for name in ("estimates.json", "benchmark.json"):
            candidate = base / name
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
    raise FileNotFoundError(f"No estimates.json found under {case_dir}")


def parse_suffix(suffix: str) -> Dict[str, str]:
    match = SUFFIX_RE.fullmatch(suffix)
    if not match:
        raise ValueError(f"Unrecognised benchmark id: {suffix}")
    return match.groupdict()


def discover_groups(base_dir: Path) -> List[str]:
    if not base_dir.exists():
        return []
    groups: List[str] = []
    for group_dir in sorted(entry for entry in base_dir.iterdir() if entry.is_dir()):
        for case_dir in group_dir.iterdir():
            if not case_dir.is_dir():
                continue
            try:
                parse_suffix(case_dir.name)
            except ValueError:
                continue
            groups.append(group_dir.name)
            break
    return groups


def collect_records(base_dir: Path, groups: Sequence[str]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for group in groups:
        group_dir = base_dir / group
        if not group_dir.exists():
            continue
        for case_dir in group_dir.iterdir():
            if not case_dir.is_dir():
                continue
            try:
                meta = parse_suffix(case_dir.name)
            except ValueError:
                continue
            try:
                estimates = load_estimates(case_dir)
            except FileNotFoundError:
                continue

            mean_nano_seconds = estimates["mean"]["point_estimate"]
            record = {
                "mode": group,
                "suite": meta["suite"],
                "scenario": meta["scenario"],
                "variant": meta["variant"],
                "selector_len": int(meta["selector_len"]),
                "avg_selector_len": float(meta["avg_selector_len"]),
                "select_ratio_pct": int(meta["select_ratio"]),
                "select_ratio": int(meta["select_ratio"]) / 100.0,
                "mean_seconds": mean_nano_seconds / 1_000_000_000.0,
                "mean_millis": mean_nano_seconds / 1_000_000.0,
                "benchmark_id": case_dir.name,
            }
            records.append(record)
    return records


def write_csv(records: Iterable[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "suite",
        "scenario",
        "variant",
        "selector_len",
        "avg_selector_len",
        "select_ratio",
        "select_ratio_pct",
        "mean_seconds",
        "mean_millis",
        "benchmark_id",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(records, key=lambda r: (r["suite"], r["variant"], r["mode"], r["selector_len"])):
            writer.writerow(row)


def normalise_name(*parts: str) -> str:
    combined = "-".join(parts)
    return re.sub(r"[^A-Za-z0-9_.-]", "_", combined)


def make_plots(
    records: Iterable[Dict[str, object]], plot_dir: Path, mode_order: Sequence[str]
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    series: Dict[Tuple[str, str, str], Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for rec in records:
        key = (rec["suite"], rec["variant"], rec["scenario"])
        length_map = series[key]
        length_map[rec["selector_len"]][rec["mode"]] = rec["mean_millis"]

    mode_order = list(mode_order) or sorted({rec["mode"] for rec in records})
    for (suite, variant, scenario), length_map in series.items():
        lengths = sorted(length_map.keys())
        if not lengths:
            continue
        fig, ax = plt.subplots()
        plotted = False
        for mode in mode_order:
            values = [length_map[length].get(mode) for length in lengths]
            if not values or any(value is None for value in values):
                continue
            ax.plot(lengths, values, marker="o", label=mode)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Selector length target")
        ax.set_ylabel("Mean time per run (ms)")
        ax.set_title(f"{suite} / {variant} / {scenario}")
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        outfile = plot_dir / f"{normalise_name(suite, variant, scenario)}.png"
        fig.savefig(outfile, dpi=120)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export row_selection_state benchmarks to CSV and plots.")
    parser.add_argument("--skip-bench", action="store_true", help="Skip running cargo bench")
    parser.add_argument(
        "--output-dir",
        default=Path("target") / "criterion",
        type=Path,
        help="Root directory where Criterion stores outputs (default: target/criterion)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Destination CSV path (default: OUTPUT_DIR/row_selection_state_summary.csv)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Directory for generated plots (default: OUTPUT_DIR/row_selection_state_plots)",
    )
    parser.add_argument(
        "--group",
        dest="groups",
        action="append",
        help="Criterion group name to include (repeatable; default: auto-detect)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to cargo bench (precede with --).",
    )
    args = parser.parse_args()

    csv_path = args.csv or args.output_dir / "row_selection_state_summary.csv"
    plot_dir = args.plot_dir or args.output_dir / "row_selection_state_plots"

    run_bench(args.skip_bench, args.extra_args)

    groups = args.groups or discover_groups(args.output_dir)
    groups = list(dict.fromkeys(groups))
    if not groups:
        raise SystemExit(f"No row_selection_state benchmark groups found under {args.output_dir}")

    records = collect_records(args.output_dir, groups)
    if not records:
        raise SystemExit(
            f"No benchmark results found under {args.output_dir} (groups: {', '.join(groups)})"
        )

    write_csv(records, csv_path)
    make_plots(records, plot_dir, groups)
    print(f"Wrote CSV to {csv_path}")
    print(f"Plots in {plot_dir} (groups: {', '.join(groups)})")


if __name__ == "__main__":
    main()
