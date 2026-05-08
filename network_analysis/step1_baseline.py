"""
step1_baseline.py
=================
Baseline comparison: frozen random reservoir vs Oja-learned weights.

Question
--------
Does Oja learning actually improve reconstruction quality compared to a fixed,
unlearned random reservoir?

Method
------
For each g in G_VALUES, n_seeds runs:
  1. Frozen baseline: same W0, no weight updates (oja_frozen=True).
     Readout fitted with ridge regression (alpha=1e-3) to avoid OLS
     trivial interpolation with n=50 features.
  2. Oja learned: weight updates active for T=80 (same as direction2).
     Readout fitted with standard OLS (consistent with direction2).

Outputs
-------
r2_baseline_vs_oja.png   — R² vs g for both conditions (line plot)
r2_difference.png        — ΔR² = R²_oja − R²_frozen per g with error bars
step1_data.csv           — g, seed, condition (frozen/oja), r2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'chaos_stability'))
sys.path.insert(0, os.path.join(_HERE, '..', 'oja_chaos'))

from rnn_models import init_W
from oja_core import simulate_oja, compute_r2

G_VALUES = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]


def _compute_r2_ridge(t, x_hist, omega_target, transient_frac=0.3, alpha=1e-3):
    """R² via ridge regression readout (Tikhonov)."""
    n_trans = int(len(t) * transient_frac)
    t_ss = t[n_trans:]
    X_ss = x_hist[n_trans:]
    if not np.all(np.isfinite(X_ss)):
        return np.nan
    target = np.sin(omega_target * t_ss)
    n_feat = X_ss.shape[1]
    c = np.linalg.solve(X_ss.T @ X_ss + alpha * np.eye(n_feat), X_ss.T @ target)
    y_pred = X_ss @ c
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    ss_res = float(np.sum((target - y_pred) ** 2))
    return (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def run_step1(n=50, n_seeds=5, params=None, T=80.0, dt=0.05, verbose=True):
    """
    Compare frozen random reservoir (ridge) vs Oja-learned (OLS) R².

    Returns DataFrame with columns: g, seed, condition, r2
    """
    if params is None:
        params = _default_params()
    omega_d = params.get('omega_drive', 1.0)

    records = []
    total = len(G_VALUES) * n_seeds * 2
    idx = 0

    for g in G_VALUES:
        for seed in range(n_seeds):
            W0 = init_W(n, g, seed=seed * 13 + 5)

            # Frozen baseline (ridge regression)
            p_frozen = dict(params, oja_frozen=True)
            t_act, x_hist, _, _, _ = simulate_oja(
                W0, p_frozen, T=T, dt=dt,
                rho_sample_every=999999, seed=seed,
            )
            r2_frozen = _compute_r2_ridge(t_act, x_hist, omega_d)

            # Oja learned (OLS, consistent with direction2)
            p_oja = dict(params, oja_frozen=False)
            t_act2, x_hist2, _, _, _ = simulate_oja(
                W0, p_oja, T=T, dt=dt,
                rho_sample_every=999999, seed=seed,
            )
            r2_oja = compute_r2(t_act2, x_hist2, omega_d, transient_frac=0.3)

            records.append(dict(g=g, seed=seed, condition='frozen', r2=float(r2_frozen)))
            records.append(dict(g=g, seed=seed, condition='oja',    r2=float(r2_oja)))

            idx += 2
            if verbose:
                print(f"  Step1 [{idx:3d}/{total}]  g={g:.2f}  "
                      f"R²_frozen={r2_frozen:.3f}  R²_oja={r2_oja:.3f}")

    return pd.DataFrame(records)


def plot_step1(df, save_dir=_HERE):
    """Save baseline vs Oja comparison plots."""
    summary = (
        df.groupby(['g', 'condition'])
        .agg(r2_mean=('r2', 'mean'), r2_std=('r2', 'std'))
        .reset_index()
    )
    frozen_s = summary[summary['condition'] == 'frozen'].sort_values('g').reset_index(drop=True)
    oja_s    = summary[summary['condition'] == 'oja'].sort_values('g').reset_index(drop=True)
    g_vals   = frozen_s['g'].values

    # Plot 1: R² vs g for both conditions
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(g_vals, frozen_s['r2_mean'], yerr=frozen_s['r2_std'],
                fmt='o-', color='steelblue', capsize=4, lw=2.0,
                label='Frozen random reservoir  (ridge, α=1e-3)')
    ax.errorbar(g_vals, oja_s['r2_mean'], yerr=oja_s['r2_std'],
                fmt='s-', color='tomato', capsize=4, lw=2.0,
                label='Oja-learned  (OLS)')
    ax.axhline(0.0, color='gray', lw=0.8, linestyle='--')
    ax.set_xlabel('initial spectral radius  g', fontsize=12)
    ax.set_ylabel('R²  (mean ± std)', fontsize=12)
    ax.set_title('Reconstruction quality: frozen random reservoir vs Oja learning\n'
                 '(T=80, sinusoidal drive, n=50 neurons)', fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(save_dir, 'r2_baseline_vs_oja.png')
    fig.savefig(p1, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p1}")

    # Plot 2: ΔR² = R²_oja − R²_frozen
    delta_mean = oja_s['r2_mean'].values - frozen_s['r2_mean'].values
    delta_std  = np.sqrt(oja_s['r2_std'].values ** 2 + frozen_s['r2_std'].values ** 2)
    colors     = ['tomato' if d > 0 else 'steelblue' for d in delta_mean]
    x_pos      = np.arange(len(g_vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos, delta_mean, yerr=delta_std, color=colors, alpha=0.75, capsize=4,
           error_kw=dict(lw=1.5, ecolor='black'))
    ax.axhline(0.0, color='black', lw=1.2, linestyle='-')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{g:.2g}" for g in g_vals], fontsize=9)
    ax.set_xlabel('initial spectral radius  g', fontsize=12)
    ax.set_ylabel('ΔR²  =  R²_oja  −  R²_frozen', fontsize=12)
    ax.set_title('Does Oja learning help reconstruction?\n'
                 '(red = Oja better, blue = frozen better)', fontsize=11)
    fig.tight_layout()
    p2 = os.path.join(save_dir, 'r2_difference.png')
    fig.savefig(p2, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p2}")

    # CSV
    csv_path = os.path.join(save_dir, 'step1_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def _default_params():
    return {
        'tau': 1.0, 'eta': 0.01, 'lambda': 0.001,
        'gain': 1.0, 'act_kind': 'tanh',
        'A': 1.0, 'omega_drive': 1.0,
    }
