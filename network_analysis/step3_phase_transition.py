"""
step3_phase_transition.py
=========================
Phase transition physics at the edge of chaos.

Question
--------
Can we observe the g=1 phase transition through physical observables
beyond the LLE — critical slowing down and avalanche statistics?

Part A — Critical slowing down
-------------------------------
For each g, compute LLE of the autonomous discrete map via power iteration.
  tau_relax = -1 / LLE  for  LLE < 0  (stable: perturbations decay)
  tau_relax = nan        for  LLE >= 0 (chaotic or critical)
At g->1 from below, LLE->0 so tau_relax->inf (critical slowing down).
Also show delta(t) decay curves for g in {0.5, 1.0, 1.5} with warmup.

Part B — Avalanche size distribution
--------------------------------------
Run discrete map for T_steps=20000. Binarize |h[i,t]| > theta=0.1.
Measure population activity A(t) = number of active neurons per step.
Fit log-log histogram to power law P(A) ~ A^{-alpha}.
Theory: alpha ~ 3/2 at g=1 (Beggs & Plenz criticality).

Outputs
-------
critical_slowing_down.png   — tau_relax vs g + delta(t) curves for 3 g values
avalanche_distributions.png — log-log P(A) for g in {0.5, 1.0, 1.5, 2.0}
power_law_exponents.png     — fitted alpha vs g
step3_data.csv              — g, seed, tau_relax, lle, alpha_fit, alpha_r2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'chaos_stability'))
sys.path.insert(0, os.path.join(_HERE, '..', 'oja_chaos'))

from rnn_models import init_W, init_h0, get_activation, run_discrete
from chaos_metrics import compute_lle_discrete

G_VALUES      = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]
G_DELTA_PLOT  = [0.5, 1.0, 1.5]
G_AVLNCH_PLOT = [0.5, 1.0, 1.5, 2.0]
AVLNCH_THETA  = 0.1   # binarisation threshold


def _perturbed_divergence(W, f, n, T_warmup=300, T_meas=400, eps=1e-6, seed=0):
    """Warmup to attractor, then track delta(t) after perturbation."""
    h0 = init_h0(n, seed=seed)
    H_warm = run_discrete(h0, W, f, T_warmup)
    h_start = H_warm[-1]
    if not np.all(np.isfinite(h_start)):
        return None, None
    d0 = np.zeros(n)
    d0[0] = eps
    H1 = run_discrete(h_start,      W, f, T_meas)
    H2 = run_discrete(h_start + d0, W, f, T_meas)
    if not np.all(np.isfinite(H1)) or not np.all(np.isfinite(H2)):
        return None, None
    steps = np.arange(T_meas + 1)
    delta = np.linalg.norm(H1 - H2, axis=1)
    return steps, delta


def _fit_power_law(A_counts, bin_centers, min_A=2):
    """Fit P(A) ~ A^{-alpha} via log-log linear regression.
    Returns (alpha, r2_fit) or (nan, nan) if insufficient data."""
    mask = (bin_centers >= min_A) & (A_counts > 0)
    if mask.sum() < 3:
        return np.nan, np.nan
    log_A = np.log(bin_centers[mask])
    log_P = np.log(A_counts[mask])
    slope, intercept = np.polyfit(log_A, log_P, 1)
    log_P_pred = slope * log_A + intercept
    ss_res = float(np.sum((log_P - log_P_pred) ** 2))
    ss_tot = float(np.sum((log_P - log_P.mean()) ** 2))
    r2_fit = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return float(-slope), float(r2_fit)


def run_step3(n=50, n_seeds=5, params=None, dt=0.05,
              T_avalanche=20000, verbose=True, skip_avalanche=False):
    """
    Part A: LLE and tau_relax for all G_VALUES x n_seeds.
    Part B: Avalanche statistics (1 seed per g for speed; skip if skip_avalanche).

    Returns DataFrame with columns: g, seed, tau_relax, lle, alpha_fit, alpha_r2
    """
    if params is None:
        params = _default_params()

    gain     = params.get('gain', 1.0)
    act_kind = params.get('act_kind', 'tanh')
    f, fp    = get_activation(act_kind, gain=gain)

    records = []
    total   = len(G_VALUES) * n_seeds
    idx     = 0

    for g in G_VALUES:
        for seed in range(n_seeds):
            W0  = init_W(n, g, seed=seed * 11 + 7)
            lle = compute_lle_discrete(W0, f, fp, n_warmup=300, n_steps=800, seed=seed)

            if np.isfinite(lle) and lle < 0:
                tau_relax = float(-1.0 / lle)
            else:
                tau_relax = np.nan

            # Avalanche (Part B) — 1 seed only
            alpha_fit = np.nan
            alpha_r2  = np.nan
            if not skip_avalanche and seed == 0:
                h0 = init_h0(n, seed=seed)
                H  = run_discrete(h0, W0, f, T_avalanche)
                if np.all(np.isfinite(H)):
                    spike = (np.abs(H) > AVLNCH_THETA).astype(float)
                    A_t   = spike.sum(axis=1)
                    A_nz  = A_t[A_t > 0].astype(int)
                    if len(A_nz) > 50:
                        bins = np.arange(1, n + 2) - 0.5
                        counts, _ = np.histogram(A_nz, bins=bins)
                        centers   = np.arange(1, n + 1).astype(float)
                        alpha_fit, alpha_r2 = _fit_power_law(counts, centers)

            records.append(dict(
                g=g, seed=seed,
                lle=float(lle) if np.isfinite(lle) else np.nan,
                tau_relax=tau_relax,
                alpha_fit=alpha_fit,
                alpha_r2=alpha_r2,
            ))
            idx += 1
            if verbose:
                print(f"  Step3A [{idx:3d}/{total}]  g={g:.2f}  "
                      f"LLE={lle:+.4f}  tau_relax={tau_relax:.1f}" if np.isfinite(tau_relax)
                      else f"  Step3A [{idx:3d}/{total}]  g={g:.2f}  "
                           f"LLE={lle:+.4f}  tau_relax=chaotic/crit")

    return pd.DataFrame(records), f


def plot_step3(df, f, n=50, params=None, save_dir=_HERE):
    """Save all Step 3 plots."""
    if params is None:
        params = _default_params()
    _plot_critical_slowing(df, f, n, params, save_dir)
    _plot_avalanche(df, f, n, save_dir)
    _plot_power_law_exponents(df, save_dir)
    csv_path = os.path.join(save_dir, 'step3_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def _plot_critical_slowing(df, f, n, params, save_dir):
    """tau_relax vs g + delta(t) curves for 3 representative g values."""
    summary = (
        df.groupby('g')
        .agg(
            tau_mean=('tau_relax', 'mean'),
            tau_std=('tau_relax', 'std'),
            lle_mean=('lle', 'mean'),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: tau_relax vs g
    ax = axes[0]
    g_vals   = summary['g'].values
    tau_mean = summary['tau_mean'].values
    tau_std  = summary['tau_std'].values
    finite   = np.isfinite(tau_mean)
    ax.errorbar(g_vals[finite], tau_mean[finite], yerr=tau_std[finite],
                fmt='o-', color='steelblue', capsize=4, lw=2.0)
    ax.axvline(1.0, linestyle='--', color='black', lw=1.0, label='g = 1  (edge of chaos)')
    ax.set_xlabel('spectral radius  g', fontsize=12)
    ax.set_ylabel('relaxation time  τ_relax  (steps)', fontsize=12)
    ax.set_title('Critical slowing down near g = 1', fontsize=11)
    ax.legend(fontsize=9)

    # Right: delta(t) curves for 3 g values
    ax = axes[1]
    colors = {'0.5': 'steelblue', '1.0': 'black', '1.5': 'tomato'}
    for g_plot in G_DELTA_PLOT:
        W0     = init_W(n, g_plot, seed=7)
        steps, delta = _perturbed_divergence(W0, f, n, seed=0)
        if steps is None:
            continue
        color = colors.get(str(g_plot), 'gray')
        ax.semilogy(steps, delta + 1e-300, lw=2.0, color=color,
                    label=f'g = {g_plot}')
    ax.axhline(1e-6, color='gray', lw=0.8, linestyle=':', label='ε = 1e-6')
    ax.set_xlabel('steps after perturbation', fontsize=12)
    ax.set_ylabel('‖δh(t)‖  (log scale)', fontsize=12)
    ax.set_title('Perturbation divergence / decay after warmup', fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle('Phase transition observable: critical slowing down', fontsize=12)
    fig.tight_layout()
    p = os.path.join(save_dir, 'critical_slowing_down.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_avalanche(df, f, n, save_dir):
    """Log-log P(A) histograms for 4 g values."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for ax_idx, g_plot in enumerate(G_AVLNCH_PLOT):
        ax  = axes_flat[ax_idx]
        W0  = init_W(n, g_plot, seed=7)
        h0  = init_h0(n, seed=0)
        H   = run_discrete(h0, W0, f, T_steps=8000)

        if not np.all(np.isfinite(H)):
            ax.set_title(f'g={g_plot}  (diverged)')
            continue

        spike = (np.abs(H) > AVLNCH_THETA).astype(float)
        A_t   = spike.sum(axis=1)
        A_nz  = A_t[A_t > 0].astype(int)

        bins    = np.arange(1, n + 2) - 0.5
        counts, _ = np.histogram(A_nz, bins=bins)
        centers = np.arange(1, n + 1).astype(float)
        mask    = counts > 0

        ax.loglog(centers[mask], counts[mask], 'o', color='steelblue',
                  alpha=0.8, markersize=4, label='data')

        # Reference power law slope -3/2
        if mask.sum() > 1:
            x_ref = centers[mask]
            y_ref = counts[mask].max() * (x_ref / x_ref.min()) ** (-1.5)
            ax.loglog(x_ref, y_ref, 'k--', lw=1.5, label='slope = -3/2')

        ax.set_xlabel('avalanche size A', fontsize=10)
        ax.set_ylabel('count', fontsize=10)
        ax.set_title(f'g = {g_plot}', fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle('Avalanche size distributions  P(A)  for discrete RNN\n'
                 '(threshold = 0.1; power law P~A⁻³/² predicted at g=1)',
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(save_dir, 'avalanche_distributions.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_power_law_exponents(df, save_dir):
    """Fitted power-law exponent alpha vs g."""
    sub = df[df['seed'] == 0].dropna(subset=['alpha_fit']).sort_values('g')
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sub['g'], sub['alpha_fit'], 'o-', color='steelblue', lw=2.0, markersize=6)
    ax.axhline(1.5, linestyle='--', color='black', lw=1.0, label='α = 3/2  (critical theory)')
    ax.axvline(1.0, linestyle=':', color='gray', lw=1.0, label='g = 1')
    ax.set_xlabel('spectral radius  g', fontsize=12)
    ax.set_ylabel('fitted power-law exponent  α', fontsize=12)
    ax.set_title('Power-law exponent of avalanche distribution vs g', fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(save_dir, 'power_law_exponents.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _default_params():
    return {
        'tau': 1.0, 'eta': 0.01, 'lambda': 0.001,
        'gain': 1.0, 'act_kind': 'tanh',
        'A': 1.0, 'omega_drive': 1.0,
    }
