#!/usr/bin/env python
"""Draw the Reacher/Cube local action-metric schematic used in the paper."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"),
)


def ellipse_xy(
    major: float,
    minor: float,
    angle_deg: float,
    *,
    n: int = 500,
):
    import numpy as np

    t = np.linspace(0.0, 2.0 * math.pi, n)
    xy = np.vstack((major * np.cos(t), minor * np.sin(t)))
    angle = math.radians(angle_deg)
    rot = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    return rot @ xy


def draw_axes(ax, xlabel: str, ylabel: str) -> None:
    lim = 2.16
    ax.annotate(
        "",
        xy=(lim, 0),
        xytext=(-lim, 0),
        arrowprops=dict(arrowstyle="->", color="#C9C9C9", lw=0.75),
        zorder=0,
    )
    ax.annotate(
        "",
        xy=(0, lim),
        xytext=(0, -lim),
        arrowprops=dict(arrowstyle="->", color="#C9C9C9", lw=0.75),
        zorder=0,
    )
    ax.text(lim + 0.05, -0.03, xlabel, ha="left", va="top", fontsize=8.2)
    ax.text(0.06, lim - 0.13, ylabel, ha="left", va="bottom", fontsize=8.2)
    ax.set_xlim(-2.42, 2.52)
    ax.set_ylim(-2.42, 2.42)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_metric_pair(
    ax,
    *,
    env_angle: float,
    model_angle: float,
    env_major: float,
    env_minor: float,
    model_major: float,
    model_minor: float,
    angle_label: str,
    stats: str,
    panel_label: str,
    title: str,
    plane_label: str,
    env_label_xy: tuple[float, float],
    model_label_xy: tuple[float, float],
) -> None:
    import matplotlib.patches as patches

    env_color = "#2F6FB0"
    model_color = "#E36C0A"
    axis_color = "#111111"

    for major, minor, angle, color, z in (
        (env_major, env_minor, env_angle, env_color, 2),
        (model_major, model_minor, model_angle, model_color, 3),
    ):
        xy = ellipse_xy(major, minor, angle)
        ax.fill(xy[0], xy[1], color=color, alpha=0.055, zorder=z)
        ax.plot(xy[0], xy[1], color=color, lw=1.55, zorder=z + 1)
        length = major * 1.02
        rad = math.radians(angle)
        ax.plot(
            [-length * math.cos(rad), length * math.cos(rad)],
            [-length * math.sin(rad), length * math.sin(rad)],
            color=color,
            lw=0.85,
            zorder=z + 2,
        )

    arc_radius = 0.68
    theta1 = min(env_angle, model_angle)
    theta2 = max(env_angle, model_angle)
    arc = patches.Arc(
        (0, 0),
        2 * arc_radius,
        2 * arc_radius,
        theta1=theta1,
        theta2=theta2,
        color=axis_color,
        lw=0.8,
        zorder=8,
    )
    ax.add_patch(arc)
    mid = math.radians((theta1 + theta2) / 2.0)
    ax.text(
        0.80 * math.cos(mid),
        0.80 * math.sin(mid),
        angle_label,
        ha="center",
        va="center",
        fontsize=8.0,
        color=axis_color,
        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.94),
        zorder=9,
    )

    ax.text(
        0.0,
        0.985,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )
    if title:
        ax.text(
            0.0,
            0.91,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color="#333333",
        )
    ax.text(
        0.02,
        0.04,
        stats,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#333333",
    )
    ax.text(
        0.98,
        0.04,
        plane_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.9,
        color="#555555",
    )
    ax.text(*env_label_xy, "Environment", color=env_color, fontsize=7.0)
    ax.text(*model_label_xy, "LeWM", color=model_color, fontsize=7.0)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[1]
    out_base = root / "figures" / "local_metric_ellipses_reacher_cube"
    out_base.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55), dpi=320)
    for ax in axes:
        draw_axes(ax, r"$a_1$", r"$a_2$")

    # For a 2D metric ellipse, axis ratio is sqrt(condition number).
    reacher_env_minor = 0.96
    reacher_env_major = reacher_env_minor * math.sqrt(1.86)
    reacher_model_minor = 0.78
    reacher_model_major = reacher_model_minor * math.sqrt(4.44)

    draw_metric_pair(
        axes[0],
        env_angle=28.0,
        model_angle=32.4,
        env_major=reacher_env_major,
        env_minor=reacher_env_minor,
        model_major=reacher_model_major,
        model_minor=reacher_model_minor,
        angle_label=r"$4.4^\circ$",
        stats=(
            r"$\kappa_{\rm env}=1.86$"
            "\n"
            r"$\kappa_{\rm model}=4.44$"
        ),
        panel_label="(a) Reacher",
        title="",
        plane_label="2D action space",
        env_label_xy=(-1.75, 0.93),
        model_label_xy=(0.98, 0.93),
    )

    draw_metric_pair(
        axes[1],
        env_angle=18.0,
        model_angle=73.2,
        env_major=1.82,
        env_minor=0.72,
        model_major=1.72,
        model_minor=0.68,
        angle_label=r"$55.2^\circ$",
        stats=(
            r"$D_G=0.585$"
            "\n"
            r"$\theta_{1:2}=42.7^\circ$"
        ),
        panel_label="(b) Cube",
        title="",
        plane_label="top-2 action slice",
        env_label_xy=(-1.86, -1.28),
        model_label_xy=(0.45, 1.54),
    )

    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.035, top=0.985, wspace=0.14)

    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.03)
    print(out_base.with_suffix(".pdf"))
    print(out_base.with_suffix(".png"))


if __name__ == "__main__":
    main()
