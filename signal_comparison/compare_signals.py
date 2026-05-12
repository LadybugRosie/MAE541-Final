"""
compare_signals.py
==================
Does rank collapse in W depend on the temporal structure of the input?

Hypothesis
----------
Oja's rule drives W toward W* ≈ (η/λ)⟨xx^T⟩.
  - Sinusoidal drive:  all neurons lock to ω → ⟨xx^T⟩ ≈ (A²/2) r rᵀ  (rank-1)
    → W* is rank-1 → self-reinforcing collapse → ρ(W) → ∞
  - White noise:       no shared frequency → ⟨xx^T⟩ ≈ σ²I (full-rank)
    → W* ≈ isotropic → no collapse
  - Quasi-periodic (k incommensurate freqs):
    ⟨xx^T⟩ has rank ≤ k → intermediate collapse

Drive types tested
------------------
  sin   : A·sin(ωt + φᵢ)                                  [rank-1 correlation]
  quasi : A/√3 · Σⱼ sin(ωⱼt + φᵢ), ω incommensurate       [rank-3 correlation]
  white : A/√2 · ξᵢ(t), ξ~N(0,1) per step                 [isotropic]
  zero  : no input                                          [autonomous baseline]

Outputs
-------
rho_vs_time_by_drive.png       — ρ(W) trajectories for all drives (4-panel)
pr_vs_time_by_drive.png        — PR(W) trajectories (4-panel)
correlation_rank_by_drive.png  — rank(⟨xx^T⟩) sin vs white at g=1.0
r2_by_drive.png                — R² vs g for all drives
final_summary.png              — final ρ and PR bar charts
comparison_data.csv            — raw trace data
"""

import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'chaos_stability'))
sys.path.insert(0, os.path.join(_HERE, '..', 'oja_chaos'))

from rnn_models import init_W, get_activation
from oja_core import compute_r2

DRIVE_TYPES = ['sin', 'quasi', 'white', 'zero']
G_VALUES    = [0.3, 0.7, 1.0, 1.5, 2.0]

_DRIVE_COLORS = {
    'sin':   'tomato',
    'quasi': 'darkorange',
    'white': 'steelblue',
    'zero':  'gray',
}
_DRIVE_LABELS = {
    'sin':   'Sinusoidal  (rank-1 ⟨xx^T⟩)',
    'quasi': 'Quasi-periodic  (rank-3 ⟨xx^T⟩)',
    'white': 'White noise  (full-rank ⟨xx^T⟩)',
    'zero':  'No input  (autonomous)',
}


# ============================================================
# W metrics
# ============================================================

def _w_metrics(W):
    """Spectral radius, rank, participation ratio, Frobenius norm."""
    sigmas = np.linalg.svd(W, compute_uv=False)
    s2     = float((sigmas ** 2).sum())
    return dict(
        rho  = float(np.abs(np.linalg.eigvals(W)).max()),
        rank = int(np.linalg.matrix_rank(W, tol=1e-3)),
        PR   = float(sigmas.sum() ** 2 / s2) if s2 > 1e-12 else 0.0,
        frob = float(np.linalg.norm(W, 'fro')),
    )


# ============================================================
# Drive factory
# ============================================================

def make_drive(kind, A, omega, n, seed=0):
    """
    Returns drive_fn(t, rng) -> (n,) array.

    Per-neuron phase offsets φᵢ are fixed from `seed` so drives are
    reproducible across g values.
    """
    phi = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, n)
    w2  = omega * 1.6180339887   # golden ratio — incommensurate with 1
    w3  = omega * 1.4142135623   # sqrt(2) — incommensurate with 1 and φ

    if kind == 'sin':
        return lambda t, rng: A * np.sin(omega * t + phi)
    elif kind == 'quasi':
        return lambda t, rng: (A / np.sqrt(3)) * (
            np.sin(omega * t + phi)
            + np.sin(w2 * t + phi)
            + np.sin(w3 * t + phi)
        )
    elif kind == 'white':
        # RMS power = A/√2 · √n, same as sinusoidal drive per neuron
        return lambda t, rng: rng.normal(0.0, A / np.sqrt(2), n)
    elif kind == 'zero':
        return lambda t, rng: np.zeros(n)
    else:
        raise ValueError(f"Unknown drive kind: {kind!r}")


# ============================================================
# Oja simulation with configurable drive
# ============================================================

def simulate_with_drive(
    W0, params, drive_fn,
    T=300.0, dt=0.05,
    sample_every=20, corr_window=200, seed=0,
):
    """
    RK4 on x, Euler on W.  Activity (RK4): τẋ = -x + f(Wx + drive(t))
    Weights (Euler): dW/dt = η(xxᵀ - diag(x²)W) - λW

    White noise uses a single draw per full step, held constant across RK4 stages.

    Parameters
    ----------
    drive_fn : callable(t, rng) -> (n,) array
    corr_window : int    number of recent steps used to estimate ⟨xx^T⟩

    Returns
    -------
    records : list of dicts with keys t, rho, rank_W, PR, frob, rank_C
    x_hist  : (n_steps, n)
    t_act   : (n_steps,)
    W_final : (n, n)
    """
    n           = W0.shape[0]
    tau         = params.get('tau', 1.0)
    eta         = params.get('eta', 0.01)
    lam         = params.get('lambda', 0.001)
    gain        = params.get('gain', 1.0)
    act_kind    = params.get('act_kind', 'tanh')

    f, _  = get_activation(act_kind, gain=gain)
    rng   = np.random.default_rng(seed)
    x     = rng.normal(0.0, 0.01, n)
    W     = W0.copy().astype(float)

    n_steps = int(round(T / dt))
    t_act   = np.arange(n_steps) * dt
    x_hist  = np.empty((n_steps, n))

    x_buffer = collections.deque(maxlen=corr_window)
    records  = []

    def _rhs(x_, drive_vec):
        return (-x_ + f(W @ x_ + drive_vec)) / tau

    for i in range(n_steps):
        t = i * dt

        # Structural sample
        if i % sample_every == 0:
            m = _w_metrics(W)
            # Short-window correlation rank
            if len(x_buffer) >= 10:
                X_buf  = np.array(x_buffer)       # (window, n)
                C_buf  = X_buf.T @ X_buf / len(x_buffer)
                rank_C = int(np.linalg.matrix_rank(C_buf, tol=1e-3))
            else:
                rank_C = n   # not enough data yet
            records.append(dict(t=t, rank_C=rank_C, **m))

        x_hist[i] = x
        x_buffer.append(x.copy())

        # Draw drive once per step (frozen across RK4 sub-stages)
        d = drive_fn(t, rng)

        k1 = _rhs(x,                    d)
        k2 = _rhs(x + 0.5 * dt * k1,   d)
        k3 = _rhs(x + 0.5 * dt * k2,   d)
        k4 = _rhs(x + dt * k3,         d)
        x_new = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.all(np.isfinite(x_new)):
            x_hist[i + 1:] = np.nan
            break

        dW = eta * (np.outer(x, x) - (x ** 2)[:, None] * W) - lam * W
        W  = W + dt * dW
        x  = x_new

    return records, x_hist, t_act, W


# ============================================================
# Main experiment
# ============================================================

def run_comparison(
    n=50, n_seeds=3, params=None,
    T=300.0, dt=0.05, T_r2=80.0,
    verbose=True,
):
    """
    For each (drive_type, g, seed):
      - Run T=300 simulation → structural metrics over time
      - Run T=80 simulation  → R² for sin(ωt) reconstruction

    Returns (trace_df, r2_df)
    """
    if params is None:
        params = _default_params()

    A     = params.get('A', 1.0)
    omega = params.get('omega_drive', 1.0)

    trace_records = []
    r2_records    = []
    total = len(DRIVE_TYPES) * len(G_VALUES) * n_seeds
    idx   = 0

    for drive_kind in DRIVE_TYPES:
        for g in G_VALUES:
            for seed in range(n_seeds):
                W0       = init_W(n, g, seed=seed * 13 + 5)
                drive_fn = make_drive(drive_kind, A, omega, n, seed=seed)

                # Long trace (T=300)
                recs, _, _, _ = simulate_with_drive(
                    W0, params, drive_fn,
                    T=T, dt=dt, sample_every=20, corr_window=200, seed=seed,
                )
                for r in recs:
                    trace_records.append(dict(drive=drive_kind, g=g, seed=seed, **r))

                # Short run (T=80) for R²
                drive_fn2 = make_drive(drive_kind, A, omega, n, seed=seed)
                _, x_hist, t_act, _ = simulate_with_drive(
                    W0, params, drive_fn2,
                    T=T_r2, dt=dt, sample_every=999999, corr_window=1, seed=seed,
                )
                r2 = compute_r2(t_act, x_hist, omega, transient_frac=0.3)
                r2_records.append(dict(drive=drive_kind, g=g, seed=seed, r2=float(r2)))

                idx += 1
                if verbose:
                    final_rho = recs[-1]['rho'] if recs else np.nan
                    final_PR  = recs[-1]['PR']  if recs else np.nan
                    print(f"  [{idx:3d}/{total}]  drive={drive_kind:6s}  g={g:.2f}  "
                          f"seed={seed}  ρ_final={final_rho:.2f}  "
                          f"PR_final={final_PR:.2f}  R²={r2:.3f}")

    return pd.DataFrame(trace_records), pd.DataFrame(r2_records)


# ============================================================
# Plotting
# ============================================================

def plot_comparison(trace_df, r2_df, save_dir=_HERE):
    _plot_rho_panels(trace_df, save_dir)
    _plot_pr_panels(trace_df, save_dir)
    _plot_correlation_rank(trace_df, save_dir)
    _plot_r2_by_drive(r2_df, save_dir)
    _plot_final_summary(trace_df, save_dir)

    csv_path = os.path.join(save_dir, 'comparison_data.csv')
    trace_df.to_csv(csv_path, index=False)
    r2_df.to_csv(os.path.join(save_dir, 'comparison_r2.csv'), index=False)
    print(f"  Saved: {csv_path}")


def _plot_rho_panels(trace_df, save_dir):
    """4-panel: ρ(W) over time for each drive type."""
    g_show   = [0.3, 1.0, 2.0]
    g_colors = {0.3: 'steelblue', 1.0: 'black', 2.0: 'tomato'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    for ax, drive_kind in zip(axes.flatten(), DRIVE_TYPES):
        sub = trace_df[trace_df['drive'] == drive_kind]
        for g in g_show:
            s = sub[sub['g'] == g]
            if s.empty:
                continue
            mean = s.groupby('t')['rho'].mean()
            std  = s.groupby('t')['rho'].std().fillna(0)
            ax.plot(mean.index, mean.values, color=g_colors[g], lw=2.0,
                    label=f'g={g}')
            ax.fill_between(mean.index,
                            mean.values - std.values,
                            mean.values + std.values,
                            color=g_colors[g], alpha=0.15)
        ax.axhline(1.0, linestyle='--', color='gray', lw=0.8, label='ρ=1 (edge)')
        ax.set_title(_DRIVE_LABELS[drive_kind], fontsize=10)
        ax.set_xlabel('time', fontsize=10)
        ax.set_ylabel('ρ(W)', fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle('Spectral radius ρ(W) under different input drives\n'
                 '(Does sinusoidal uniquely cause ρ → ∞?)', fontsize=12)
    fig.tight_layout()
    p = os.path.join(save_dir, 'rho_vs_time_by_drive.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_pr_panels(trace_df, save_dir):
    """4-panel: PR(W) over time for each drive type."""
    g_show   = [0.3, 1.0, 2.0]
    g_colors = {0.3: 'steelblue', 1.0: 'black', 2.0: 'tomato'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    for ax, drive_kind in zip(axes.flatten(), DRIVE_TYPES):
        sub = trace_df[trace_df['drive'] == drive_kind]
        for g in g_show:
            s = sub[sub['g'] == g]
            if s.empty:
                continue
            mean = s.groupby('t')['PR'].mean()
            std  = s.groupby('t')['PR'].std().fillna(0)
            ax.plot(mean.index, mean.values, color=g_colors[g], lw=2.0,
                    label=f'g={g}')
            ax.fill_between(mean.index,
                            mean.values - std.values,
                            mean.values + std.values,
                            color=g_colors[g], alpha=0.15)
        ax.set_title(_DRIVE_LABELS[drive_kind], fontsize=10)
        ax.set_xlabel('time', fontsize=10)
        ax.set_ylabel('participation ratio  PR', fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle('Effective dimensionality PR(W) under different input drives\n'
                 '(Does sinusoidal uniquely collapse dimensionality?)', fontsize=12)
    fig.tight_layout()
    p = os.path.join(save_dir, 'pr_vs_time_by_drive.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_correlation_rank(trace_df, save_dir):
    """
    2-panel side-by-side: rank(⟨xx^T⟩) vs time for sin and white, at g=1.0.
    This is the direct mechanistic test.
    """
    g_target = 1.0
    drives   = ['sin', 'white']
    titles   = [_DRIVE_LABELS['sin'], _DRIVE_LABELS['white']]
    colors   = ['tomato', 'steelblue']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, drive_kind, title, color in zip(axes, drives, titles, colors):
        sub = trace_df[(trace_df['drive'] == drive_kind) &
                       (trace_df['g'] == g_target)]
        if sub.empty:
            ax.set_title(f'{title} (no data)')
            continue
        mean = sub.groupby('t')['rank_C'].mean()
        std  = sub.groupby('t')['rank_C'].std().fillna(0)
        ax.plot(mean.index, mean.values, color=color, lw=2.5, label='rank(⟨xx^T⟩)')
        ax.fill_between(mean.index,
                        mean.values - std.values,
                        mean.values + std.values,
                        color=color, alpha=0.2)
        ax.axhline(1.0, linestyle='--', color='black', lw=1.0, label='rank = 1')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('time', fontsize=11)
        ax.set_ylabel('rank(⟨xx^T⟩)  over rolling 200-step window', fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle(f'Rank of activity correlation matrix ⟨xx^T⟩ over time  (g={g_target})\n'
                 'Sinusoidal drive → rank collapses to 1;  white noise → stays full-rank',
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(save_dir, 'correlation_rank_by_drive.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_r2_by_drive(r2_df, save_dir):
    """R² vs g for all drive types."""
    summary = (
        r2_df.groupby(['drive', 'g'])
        .agg(r2_mean=('r2', 'mean'), r2_std=('r2', 'std'))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    markers = {'sin': 'o', 'quasi': 's', 'white': '^', 'zero': 'D'}
    for drive_kind in DRIVE_TYPES:
        sub = summary[summary['drive'] == drive_kind].sort_values('g')
        if sub.empty:
            continue
        ax.errorbar(
            sub['g'], sub['r2_mean'], yerr=sub['r2_std'],
            fmt=markers[drive_kind] + '-',
            color=_DRIVE_COLORS[drive_kind], capsize=4, lw=2.0,
            label=_DRIVE_LABELS[drive_kind],
        )
    ax.axhline(0.0, color='gray', lw=0.8, linestyle='--')
    ax.set_xlabel('initial spectral radius  g', fontsize=12)
    ax.set_ylabel('R²  (Fourier reconstruction at T=80)', fontsize=12)
    ax.set_title('Fourier reconstruction quality by drive type\n'
                 '(Does white noise preserve R² where sinusoidal fails?)', fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(save_dir, 'r2_by_drive.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_final_summary(trace_df, save_dir):
    """2×2: final ρ and final PR for each (drive, g)."""
    # Get final time point per (drive, g, seed)
    idx_max = trace_df.groupby(['drive', 'g', 'seed'])['t'].idxmax()
    final   = trace_df.loc[idx_max]
    summary = (
        final.groupby(['drive', 'g'])
        .agg(rho_mean=('rho', 'mean'), rho_std=('rho', 'std'),
             PR_mean=('PR', 'mean'), PR_std=('PR', 'std'))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_pos  = np.arange(len(G_VALUES))
    width  = 0.18
    offsets = np.linspace(-1.5, 1.5, len(DRIVE_TYPES)) * width

    for metric, ax, ylabel, title in [
        ('rho', axes[0], 'final ρ(W)', 'Final spectral radius  ρ(W)'),
        ('PR',  axes[1], 'final PR(W)', 'Final participation ratio  PR(W)'),
    ]:
        for i, drive_kind in enumerate(DRIVE_TYPES):
            sub = summary[summary['drive'] == drive_kind].sort_values('g').reset_index(drop=True)
            if sub.empty:
                continue
            vals = sub[f'{metric}_mean'].values
            errs = sub[f'{metric}_std'].values
            ax.bar(x_pos + offsets[i], vals, width, yerr=errs,
                   color=_DRIVE_COLORS[drive_kind], alpha=0.8, capsize=3,
                   label=drive_kind)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'g={g}' for g in G_VALUES], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle('Final W structure after T=300 of Oja learning: effect of drive type',
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(save_dir, 'final_summary.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _default_params():
    return {
        'tau': 1.0, 'eta': 0.01, 'lambda': 0.001,
        'gain': 1.0, 'act_kind': 'tanh',
        'A': 1.0, 'omega_drive': 1.0,
    }
