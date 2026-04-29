"""run_terminal_decoder_tests_and_plots.py

Helper script for running the triangular color code terminal boundary decoder
Has several useful benchmark settings and saving summary tables and plots
Code does: 
1. Round sweep at fixed p_data = p_meas
2. Isolated-noise diagnostics:
     * p_data = 0, p_meas > 0
     * p_meas = 0, p_data > 0
3. Optional physical-noise sweep for several round counts

Outputs a CSV summary and PNG plots into the requested output folder
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import matplotlib.pyplot as plt
import numpy as np

from stim_triangular_color_code_terminal_decoder import run_stim_z_memory_experiment


def binomial_standard_error(num_failures: int, shots: int) -> float:
    if shots <= 0:
        raise ValueError("shots must be positive")
    p_hat = num_failures / shots
    return float(np.sqrt(p_hat * (1.0 - p_hat) / shots))


def run_case(rounds: int, p_data: float, p_meas: float, shots: int) -> Dict[str, float]:
    results = run_stim_z_memory_experiment(
        rounds=rounds,
        p_data=p_data,
        p_meas=p_meas,
        shots=shots,
        add_detectors=True,
    )
    num_failures = int(results["num_failures"])
    failure_rate = float(results["failure_rate"])
    stderr = binomial_standard_error(num_failures, shots)
    return {
        "rounds": rounds,
        "p_data": p_data,
        "p_meas": p_meas,
        "shots": shots,
        "num_failures": num_failures,
        "failure_rate": failure_rate,
        "stderr": stderr,
    }


def save_csv(rows: Sequence[Dict[str, float]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows provided to save_csv")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_round_sweep_plot(rows: Sequence[Dict[str, float]], outpath: Path) -> None:

    rounds = [row["rounds"] for row in rows]
    rates = [100.0 * row["failure_rate"] for row in rows]
    errs = [100.0 * row["stderr"] for row in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(rounds, rates, yerr=errs, marker="o", linewidth=1.5, capsize=4)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical-Z failure rate (%)")
    ax.set_title("Triangular color code: equal-noise round sweep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_isolated_noise_plot(rows: Sequence[Dict[str, float]], outpath: Path) -> None:

    labels = []
    rates = []
    errs = []
    for row in rows:
        if row["p_data"] == 0.0 and row["p_meas"] > 0.0:
            labels.append("data=0, meas>0")
        elif row["p_meas"] == 0.0 and row["p_data"] > 0.0:
            labels.append("meas=0, data>0")
        else:
            labels.append(f"pd={row['p_data']}, pm={row['p_meas']}")
        rates.append(100.0 * row["failure_rate"])
        errs.append(100.0 * row["stderr"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x, rates, yerr=errs, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Logical-Z failure rate (%)")
    ax.set_title("Triangular color code: isolated-noise diagnostics")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_noise_sweep_plot(rows: Sequence[Dict[str, float]], outpath: Path) -> None:

    rounds_values = sorted({int(row["rounds"]) for row in rows})

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for rounds in rounds_values:
        sub = [row for row in rows if int(row["rounds"]) == rounds]
        p_vals = [row["p_data"] for row in sub]
        rates = [100.0 * row["failure_rate"] for row in sub]
        errs = [100.0 * row["stderr"] for row in sub]
        ax.errorbar(p_vals, rates, yerr=errs, marker="o", linewidth=1.5, capsize=3, label=f"{rounds} rounds")

    ax.set_xlabel("Physical noise probability")
    ax.set_ylabel("Logical-Z failure rate (%)")
    ax.set_title("Triangular color code: logical failure vs physical noise")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run terminal-boundary color-code benchmarks and save plots.")
    parser.add_argument("--outdir", type=str, default="terminal_decoder_results", help="Directory for CSV/plot outputs.")
    parser.add_argument("--shots", type=int, default=100_000, help="Shots per benchmark point.")
    parser.add_argument("--rounds", type=int, nargs="+", default=[3, 5, 7], help="Round counts for round and noise sweeps.")
    parser.add_argument(
        "--noise-values",
        type=float,
        nargs="+",
        default=[0.0005, 0.001, 0.002, 0.005],
        help="Physical noise values for the optional p sweep.",
    )
    parser.add_argument(
        "--skip-noise-sweep",
        action="store_true",
        help="Skip the physical-noise sweep and only run the round sweep + isolated tests.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, float]] = []

    # 1) equal noise round sweep at p_data = p_meas = 0.001
    round_rows: List[Dict[str, float]] = []
    for rounds in args.rounds:
        row = run_case(rounds=rounds, p_data=0.001, p_meas=0.001, shots=args.shots)
        row["experiment"] = "equal_noise_round_sweep"
        round_rows.append(row)
        all_rows.append(row)
        print(
            f"[round sweep] rounds={rounds}, p_data=0.001, p_meas=0.001, "
            f"failures={row['num_failures']}/{args.shots}, rate={100*row['failure_rate']:.4f}%"
        )
    save_csv(round_rows, outdir / "round_sweep_summary.csv")
    make_round_sweep_plot(round_rows, outdir / "round_sweep.png")

    # 2. isolated noise diagnostics at the middle round count when available
    diag_rounds = args.rounds[len(args.rounds) // 2]
    isolated_rows: List[Dict[str, float]] = []
    for p_data, p_meas in [(0.0, 0.001), (0.001, 0.0)]:
        row = run_case(rounds=diag_rounds, p_data=p_data, p_meas=p_meas, shots=args.shots)
        row["experiment"] = "isolated_noise"
        isolated_rows.append(row)
        all_rows.append(row)
        print(
            f"[isolated] rounds={diag_rounds}, p_data={p_data}, p_meas={p_meas}, "
            f"failures={row['num_failures']}/{args.shots}, rate={100*row['failure_rate']:.4f}%"
        )
    save_csv(isolated_rows, outdir / "isolated_noise_summary.csv")
    make_isolated_noise_plot(isolated_rows, outdir / "isolated_noise.png")

    # 3. optional physical noise sweep with p_data = p_meas
    if not args.skip_noise_sweep:
        sweep_rows: List[Dict[str, float]] = []
        for rounds in args.rounds:
            for p in args.noise_values:
                row = run_case(rounds=rounds, p_data=p, p_meas=p, shots=args.shots)
                row["experiment"] = "equal_noise_p_sweep"
                sweep_rows.append(row)
                all_rows.append(row)
                print(
                    f"[p sweep] rounds={rounds}, p={p}, failures={row['num_failures']}/{args.shots}, "
                    f"rate={100*row['failure_rate']:.4f}%"
                )
        save_csv(sweep_rows, outdir / "noise_sweep_summary.csv")
        make_noise_sweep_plot(sweep_rows, outdir / "noise_sweep.png")

    save_csv(all_rows, outdir / "all_results_summary.csv")
    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
