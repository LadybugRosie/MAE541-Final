"""
Generate a publication-style demonstration of Girko's circular law.

For an N x N random matrix with independent entries W_ij ~ N(0, g^2 / N),
the eigenvalue cloud converges to the uniform disk |z| <= g.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


N = 900
G = 1.0
SEED = 541

OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "girko_circular_law.png"
PDF_PATH = OUT_DIR / "girko_circular_law.pdf"


def main() -> None:
    rng = np.random.default_rng(SEED)
    W = rng.normal(loc=0.0, scale=G / np.sqrt(N), size=(N, N))
    eigvals = np.linalg.eigvals(W)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbf8")

    disk = Circle(
        (0, 0),
        G,
        facecolor="#d7e8f7",
        edgecolor="#1f4e79",
        linewidth=2.0,
        alpha=0.38,
        zorder=0,
    )
    ax.add_patch(disk)

    ax.scatter(
        eigvals.real,
        eigvals.imag,
        s=8,
        c="#23395b",
        alpha=0.56,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    ax.axhline(0, color="#8b8b8b", lw=0.8, ls=":", zorder=1)
    ax.axvline(0, color="#8b8b8b", lw=0.8, ls=":", zorder=1)

    ax.set_aspect("equal", adjustable="box")
    lim = 1.17 * G
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\operatorname{Re}(\lambda)$")
    ax.set_ylabel(r"$\operatorname{Im}(\lambda)$")
    ax.annotate(
        r"radius $g$",
        xy=(G / np.sqrt(2), G / np.sqrt(2)),
        xytext=(0.47, 1.08),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#1f4e79"},
        color="#1f4e79",
        ha="center",
        va="bottom",
    )

    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.9)

    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    main()
