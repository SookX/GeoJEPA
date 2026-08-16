#!/usr/bin/env python
"""Plot Reacher environment-vs-model anisotropy from per-state geometry JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-pdf", default=None)
    parser.add_argument("--title", default="Reacher action anisotropy")
    return parser.parse_args()


def main() -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    input_json = Path(args.input_json)
    output_png = Path(args.output_png)
    output_pdf = Path(args.output_pdf) if args.output_pdf else None

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    rows = payload.get("per_state")
    if not rows:
        raise SystemExit(f"{input_json} does not contain per_state rows")

    k_env = np.asarray([float(row["kappa_env"]) for row in rows], dtype=np.float64)
    k_model = np.asarray([float(row["kappa_model"]) for row in rows], dtype=np.float64)
    log_env = np.log(k_env)
    log_model = np.log(k_model)
    signed_bias = log_model - log_env

    frac_exaggerated = float((k_model > k_env).mean())
    med_env = float(np.median(k_env))
    med_model = float(np.median(k_model))
    med_bias = float(np.median(signed_bias))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=220)
    ax = axes[0]
    ax.scatter(log_env, log_model, s=14, alpha=0.58, color="#2864A6", edgecolors="none")
    low = min(float(log_env.min()), float(log_model.min()), 0.0)
    high = max(float(np.percentile(log_env, 99.5)), float(np.percentile(log_model, 99.5)))
    pad = 0.08 * max(high - low, 1e-6)
    low -= pad
    high += pad
    ax.plot([low, high], [low, high], color="#1F1F1F", linewidth=1.1, linestyle="--")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(r"$\log \kappa_{\mathrm{env}}$")
    ax.set_ylabel(r"$\log \kappa_{\mathrm{model}}$")
    ax.set_title("(a) State-wise eccentricity")
    ax.text(
        0.04,
        0.96,
        "\n".join(
            [
                rf"{100.0 * frac_exaggerated:.1f}% above diagonal",
                rf"median $\kappa_{{env}}={med_env:.2f}$",
                rf"median $\kappa_{{model}}={med_model:.2f}$",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.35", "alpha": 0.94},
    )

    ax = axes[1]
    bins = np.linspace(
        float(np.percentile(signed_bias, 1)),
        float(np.percentile(signed_bias, 99)),
        28,
    )
    ax.hist(signed_bias, bins=bins, color="#D07A32", alpha=0.86, edgecolor="white", linewidth=0.6)
    ax.axvline(0.0, color="#1F1F1F", linewidth=1.1, linestyle="--")
    ax.axvline(med_bias, color="#8B2D20", linewidth=1.4)
    ax.set_xlabel(r"$\log(\kappa_{\mathrm{model}} / \kappa_{\mathrm{env}})$")
    ax.set_ylabel("states")
    ax.set_title("(b) Eccentricity bias")
    ax.text(
        0.96,
        0.96,
        rf"median = {med_bias:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.35", "alpha": 0.94},
    )

    fig.suptitle(args.title, y=1.02, fontsize=11)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    if output_pdf:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")

    summary = {
        "input_json": str(input_json),
        "output_png": str(output_png),
        "output_pdf": str(output_pdf) if output_pdf else None,
        "num_states": int(k_env.size),
        "fraction_kappa_model_gt_env": frac_exaggerated,
        "median_kappa_env": med_env,
        "median_kappa_model": med_model,
        "median_signed_log_kappa_model_over_env": med_bias,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
