#!/usr/bin/env python
"""Draw a schematic local action-space geometry comparison for Reacher."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="stablewm_home/checkpoints/reacher_workshop")
    parser.add_argument("--basename", default="reacher_local_geometry_ellipse")
    parser.add_argument("--kappa-env", type=float, default=1.86)
    parser.add_argument("--kappa-model", type=float, default=4.44)
    parser.add_argument("--angle-env-deg", type=float, default=31.0)
    parser.add_argument("--angle-offset-deg", type=float, default=4.4)
    return parser.parse_args()


def ellipse_points(kappa: float, angle_deg: float, *, scale: float = 1.0, n: int = 400):
    import numpy as np

    # For a metric ellipse x^T G x = c with condition number kappa, the
    # principal-axis radius ratio is sqrt(kappa).
    major = scale * math.sqrt(kappa)
    minor = scale
    t = np.linspace(0.0, 2.0 * math.pi, n)
    pts = np.stack([major * np.cos(t), minor * np.sin(t)], axis=0)
    angle = math.radians(angle_deg)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    out = rot @ pts
    return out[0], out[1], major, minor


def main() -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta_env = args.angle_env_deg
    theta_model = args.angle_env_deg + args.angle_offset_deg
    x_env, y_env, major_env, _minor_env = ellipse_points(args.kappa_env, theta_env, scale=1.0)
    x_model, y_model, major_model, _minor_model = ellipse_points(args.kappa_model, theta_model, scale=1.0)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4.7, 3.9), dpi=260)
    env_color = "#276FBF"
    model_color = "#B64A3A"
    axis_color = "#333333"

    ax.plot(x_env, y_env, color=env_color, linewidth=1.8, label=rf"Environment, $\kappa_{{env}}\approx{args.kappa_env:.2f}$")
    ax.plot(x_model, y_model, color=model_color, linewidth=1.8, label=rf"LeWM, $\kappa_{{model}}\approx{args.kappa_model:.2f}$")

    for angle_deg, length, color, linestyle in [
        (theta_env, major_env * 1.07, env_color, "-"),
        (theta_model, major_model * 1.04, model_color, "-"),
    ]:
        angle = math.radians(angle_deg)
        ax.plot(
            [-length * math.cos(angle), length * math.cos(angle)],
            [-length * math.sin(angle), length * math.sin(angle)],
            color=color,
            linewidth=1.0,
            linestyle=linestyle,
            alpha=0.72,
        )

    arc_radius = 0.72
    arc = Arc(
        (0, 0),
        2 * arc_radius,
        2 * arc_radius,
        angle=0,
        theta1=theta_env,
        theta2=theta_model,
        color=axis_color,
        linewidth=1.0,
    )
    ax.add_patch(arc)
    mid = math.radians((theta_env + theta_model) / 2.0)
    ax.text(
        0.92 * math.cos(mid),
        0.92 * math.sin(mid),
        r"$4.4^\circ$",
        ha="left",
        va="bottom",
        color=axis_color,
    )

    ax.axhline(0, color="#BBBBBB", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=0)
    ax.scatter([0], [0], s=12, color=axis_color, zorder=4)

    lim = max(np.abs(x_model).max(), np.abs(y_model).max(), np.abs(x_env).max(), np.abs(y_env).max()) * 1.17
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$a_1$")
    ax.set_ylabel(r"$a_2$", rotation=0, labelpad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", frameon=False, handlelength=2.6, fontsize=8.8)

    ax.text(
        0.5,
        -0.105,
        "LeWM preserves the dominant action direction but exaggerates local anisotropy.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.7,
    )

    fig.tight_layout(pad=0.9)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{args.basename}.{ext}", bbox_inches="tight")
    print(output_dir / f"{args.basename}.png")
    print(output_dir / f"{args.basename}.pdf")
    print(output_dir / f"{args.basename}.svg")


if __name__ == "__main__":
    main()
