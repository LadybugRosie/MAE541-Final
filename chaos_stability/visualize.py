"""
visualize.py
============
All seven plot types for the chaos / stability experiment.
Every function saves a PNG and closes the figure.
"""

import numpy as np
import matplotlib.pyplot as plt

from rnn_models import (
    init_W, init_h0, get_activation, run_discrete, run_continuous,
    ALL_ACTIVATIONS,
)
from chaos_metrics import trajectory_divergence, autocorrelation, power_spectral_density

# three canonical regimes used across qualitative plots
_G_EXAMPLE  = [0.5, 1.0, 2.0]
_G_LABELS   = ["g=0.5 (stable)", "g=1.0 (edge)", "g=2.0 (chaotic)"]
_G_COLORS   = ["steelblue", "darkorange", "crimson"]


# ============================================================
# 1. LLE vs g  — map and flow comparison
# ============================================================

def plot_lle_vs_g(df_A, save_path="lle_vs_g.png"):
    """
    Line plot of LLE (mean ± std) vs spectral radius g for both formulations.
    Discrete units: nats/step.  Continuous units: nats/time.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for col, color, label in [
        ("lle_map",  "steelblue", "Discrete map  (nats / step)"),
        ("lle_flow", "tomato",    "Continuous flow  (nats / time)"),
    ]:
        mu  = df_A[f"{col}_mean"].values
        std = df_A[f"{col}_std"].values
        g   = df_A["g"].values
        ax.plot(g, mu, color=color, lw=2.0, label=label)
        ax.fill_between(g, mu - std, mu + std, color=color, alpha=0.18)

    ax.axhline(0.0, linestyle="--", color="black", lw=0.9,
               label="LLE = 0  (edge of chaos)")
    ax.axvline(1.0, linestyle=":",  color="gray",  lw=0.9,
               label="g = 1  (theory)")
    ax.set_xlabel("spectral radius  g", fontsize=12)
    ax.set_ylabel("largest Lyapunov exponent  λ₁", fontsize=12)
    ax.set_title("Order-to-chaos transition: discrete map vs. continuous flow",
                 fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# 2. Phase diagram — LLE in (g, activation) space
# ============================================================

def plot_phase_diagram(df_B, save_path="phase_diagram.png"):
    """
    Two heatmaps (map / flow) of median LLE over (g, activation).
    """
    g_vals = sorted(df_B["g"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, col, title in zip(
        axes,
        ["lle_map_mean", "lle_flow_mean"],
        ["Discrete map", "Continuous flow"],
    ):
        pivot = (df_B
                 .pivot_table(values=col, index="activation",
                              columns="g", aggfunc="mean")
                 .reindex(ALL_ACTIVATIONS))
        im = ax.imshow(pivot.values, aspect="auto", origin="upper",
                       cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        ax.set_xticks(range(len(g_vals)))
        ax.set_xticklabels([f"{v:.2g}" for v in g_vals],
                            rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(ALL_ACTIVATIONS)))
        ax.set_yticklabels(ALL_ACTIVATIONS, fontsize=9)
        ax.set_xlabel("g")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="LLE")

    fig.suptitle("Phase diagram: LLE in (g, activation) space", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# 3. Trajectory divergence
# ============================================================

def plot_trajectory_divergence(n=50, act_kind="tanh", gain=1.0, tau=1.0,
                                g_vals=None, T_steps=400, dt=0.05,
                                seed=42, save_path="trajectory_divergence.png"):
    """
    Log-linear plot of ‖δh(t)‖ for stable / edge / chaotic regimes,
    side-by-side for map and flow.
    """
    if g_vals is None:
        g_vals = _G_EXAMPLE
    f, fp = get_activation(act_kind, gain=gain)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mode, title in zip(
        axes, ["discrete", "continuous"], ["Discrete map", "Continuous flow"]
    ):
        for g, color, label in zip(g_vals, _G_COLORS, _G_LABELS):
            W = init_W(n, g, seed=seed)
            t_axis, delta = trajectory_divergence(
                W, f, tau, mode, T_steps=T_steps, dt=dt
            )
            if t_axis is None:
                continue
            valid = np.isfinite(delta) & (delta > 0)
            ax.semilogy(t_axis[valid], delta[valid],
                        color=color, lw=1.5, label=label)

        ax.set_xlabel("step" if mode == "discrete" else "time")
        ax.set_ylabel("‖δh(t)‖")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Trajectory divergence  (ε = 1 × 10⁻⁶ perturbation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# 4. Autocorrelation
# ============================================================

def plot_autocorrelation(n=50, act_kind="tanh", gain=1.0, tau=1.0,
                          g_vals=None, T_steps=3000, dt=0.05,
                          max_lag=200, seed=42,
                          save_path="autocorrelation.png"):
    """
    Normalised autocorrelation C(lag) for stable / edge / chaotic regimes.
    """
    if g_vals is None:
        g_vals = _G_EXAMPLE
    f, fp = get_activation(act_kind, gain=gain)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mode, title in zip(
        axes, ["discrete", "continuous"], ["Discrete map", "Continuous flow"]
    ):
        for g, color, label in zip(g_vals, _G_COLORS, _G_LABELS):
            W  = init_W(n, g, seed=seed)
            h0 = init_h0(n, seed=seed)

            if mode == "discrete":
                H  = run_discrete(h0, W, f, T_steps)
                x  = H[min(500, T_steps // 5):, 0]
            else:
                t, H = run_continuous(h0, W, f, tau, T_steps * dt, dt)
                if H is None:
                    continue
                n_warmup = max(1, len(H) // 5)
                x = H[n_warmup:, 0]

            if not np.all(np.isfinite(x)):
                continue
            corr = autocorrelation(x, max_lag=min(max_lag, len(x) // 2))
            ax.plot(np.arange(len(corr)), corr,
                    color=color, lw=1.5, label=label)

        ax.axhline(0.0, linestyle="--", color="gray", lw=0.8)
        ax.set_xlabel("lag")
        ax.set_ylabel("C(lag)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Autocorrelation across dynamical regimes", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# 5. Power spectral density
# ============================================================

def plot_psd(n=50, act_kind="tanh", gain=1.0, tau=1.0,
             g_vals=None, T_steps=5000, dt=0.05,
             seed=42, save_path="power_spectrum.png"):
    """
    PSD for stable / edge / chaotic regimes, side-by-side for map and flow.
    """
    if g_vals is None:
        g_vals = _G_EXAMPLE
    f, fp = get_activation(act_kind, gain=gain)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mode, title in zip(
        axes, ["discrete", "continuous"], ["Discrete map", "Continuous flow"]
    ):
        for g, color, label in zip(g_vals, _G_COLORS, _G_LABELS):
            W  = init_W(n, g, seed=seed)
            h0 = init_h0(n, seed=seed)

            if mode == "discrete":
                H  = run_discrete(h0, W, f, T_steps)
                n_warmup = min(500, T_steps // 5)
                x  = H[n_warmup:, 0]
                dt_sig = 1.0
            else:
                t, H = run_continuous(h0, W, f, tau, T_steps * dt, dt)
                if H is None:
                    continue
                n_warmup = max(1, len(H) // 5)
                x = H[n_warmup:, 0]
                dt_sig = dt

            if not np.all(np.isfinite(x)) or x.std() < 1e-12:
                continue
            freqs, Pxx = power_spectral_density(x, dt=dt_sig)
            ax.semilogy(freqs, Pxx + 1e-14,
                        color=color, lw=1.0, alpha=0.85, label=label)

        ax.set_xlabel("frequency")
        ax.set_ylabel("PSD")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Power spectral density across dynamical regimes", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# 6. Network-size dependence
# ============================================================

def plot_n_dependence(df_C, save_path="n_dependence.png"):
    """
    LLE vs g for each n; expect the critical boundary to sharpen as n → ∞.
    """
    n_vals = sorted(df_C["n"].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_vals)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in zip(
        axes,
        ["lle_map_mean", "lle_flow_mean"],
        ["Discrete map", "Continuous flow"],
    ):
        for nv, color in zip(n_vals, colors):
            sub = df_C[df_C["n"] == nv]
            ax.plot(sub["g"], sub[col], color=color, lw=1.8, label=f"n={nv}")
        ax.axhline(0.0, linestyle="--", color="black", lw=0.8)
        ax.axvline(1.0, linestyle=":",  color="gray",  lw=0.8)
        ax.set_xlabel("g");  ax.set_ylabel("LLE")
        ax.set_title(title); ax.legend(fontsize=8)

    fig.suptitle("Phase-boundary sharpening with network size n", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# 7. τ sweep  (continuous flow only)
# ============================================================

def plot_tau_sweep(df_D, save_path="tau_sweep.png"):
    """
    LLE vs g for each τ; expect g_crit to shift with τ.
    """
    tau_vals = sorted(df_D["tau"].unique())
    colors   = plt.cm.plasma(np.linspace(0.1, 0.9, len(tau_vals)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for tau, color in zip(tau_vals, colors):
        sub = df_D[df_D["tau"] == tau]
        ax.plot(sub["g"], sub["lle_flow_mean"],
                color=color, lw=1.8, label=f"τ={tau}")

    ax.axhline(0.0, linestyle="--", color="black", lw=0.8)
    ax.axvline(1.0, linestyle=":",  color="gray",  lw=0.8)
    ax.set_xlabel("g", fontsize=12)
    ax.set_ylabel("LLE (continuous flow)  nats / time", fontsize=12)
    ax.set_title("Effect of time constant τ on chaos threshold", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Bonus: sample trajectories in each regime
# ============================================================

def plot_sample_trajectories(n=50, act_kind="tanh", gain=1.0, tau=1.0,
                              g_vals=None, T_steps=300, dt=0.05,
                              seed=42, save_path="sample_trajectories.png"):
    """
    Raw activity of the first three neurons for stable / edge / chaotic g.
    """
    if g_vals is None:
        g_vals = _G_EXAMPLE
    labels = _G_LABELS
    f, fp  = get_activation(act_kind, gain=gain)

    fig, axes = plt.subplots(len(g_vals), 2,
                              figsize=(14, 3.2 * len(g_vals)), squeeze=False)
    for row, (g, label) in enumerate(zip(g_vals, labels)):
        W  = init_W(n, g, seed=seed)
        h0 = init_h0(n, seed=seed)

        # discrete
        H_d = run_discrete(h0, W, f, T_steps)
        axes[row, 0].plot(H_d[:, :3])
        axes[row, 0].set_title(f"Discrete map — {label}")
        axes[row, 0].set_xlabel("step")
        axes[row, 0].set_ylabel("h")

        # continuous
        t_c, H_c = run_continuous(h0, W, f, tau, T_steps * dt, dt)
        if H_c is not None:
            axes[row, 1].plot(t_c, H_c[:, :3])
        axes[row, 1].set_title(f"Continuous flow — {label}")
        axes[row, 1].set_xlabel("time")
        axes[row, 1].set_ylabel("h")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
