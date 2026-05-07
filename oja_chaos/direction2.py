"""
direction2.py
=============
Direction 2: Edge-of-chaos Fourier reconstruction.

Question
--------
Is reconstruction quality (R²) maximized when the network's autonomous
largest Lyapunov exponent (LLE) is near zero (edge of chaos)?

Method  (single-phase)
----------------------
Sweep initial spectral radius g. For each g:
  1. Compute LLE of W₀ (initial autonomous map, before any learning).
  2. Run the full Oja-driven network for T=80 time units.
     T=80 is the "sweet spot":
       - Long enough that transients are gone (>> tau = 1).
       - Short enough that the network is not fully saturated by Oja growth
         (ρ(W) ~ 10–20 at t=80, activity still has meaningful structure).
  3. Fit a linear readout to reconstruct sin(ω_target · t) from the activity
     history of the FULL T=80 window (transient_frac=0.3, so fit uses
     t ∈ [0.3T, T]).

R² here measures *learning quality during Oja adaptation*: how consistently
does the evolving network track the target sinusoid while W is still changing?
This is sensitive to g because:
  - Too stable (g << 1):  weak recurrent amplification, slow learning,
    limited activity variation → moderate R².
  - Near edge (g ≈ 0.9):  Oja reinforces drive correlation without saturating;
    rich but coherent activity → peak R².
  - Too chaotic (g > 1.5): large initial g drives rapid Oja growth and early
    activity saturation → non-stationary, noisy readout → low R².

Note: LLE of W_initial and W_final are both computed for the *autonomous*
discrete map (no drive), same formulation as chaos_stability/ sweep_A.

Outputs
-------
r2_vs_g.png               — mean R² per initial g (bar chart with error bars)
r2_vs_initial_lle.png     — R² vs autonomous LLE of W₀ before any learning
r2_vs_final_lle.png       — R² vs autonomous LLE of W after T=80 of Oja learning
lle_initial_vs_final.png  — scatter: initial LLE vs post-learning LLE
direction2_data.csv       — raw (g, seed, lle_initial, lle_final, r2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'chaos_stability'))
sys.path.insert(0, _HERE)

from rnn_models import init_W, get_activation
from chaos_metrics import compute_lle_discrete
from oja_core import simulate_oja, compute_r2

G_VALUES = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]


def run_direction2(
    n=50, n_seeds=5,
    params=None, T=80.0, dt=0.05,
    omega_target=None,
    disc_kwargs=None, verbose=True,
):
    """
    Sweep g values; for each compute LLE (before/after Oja) and R².

    Parameters
    ----------
    T            : simulation duration; default 80 gives a good signal window
    omega_target : target frequency for the readout (defaults to omega_drive)

    Returns
    -------
    DataFrame with columns:
        g, seed, lle_initial, lle_final, rho_initial, rho_final, r2
    """
    if params is None:
        params = _default_params()
    disc_kwargs = disc_kwargs or dict(n_warmup=200, n_steps=500)

    omega_d = params.get('omega_drive', 1.0)
    omega_t = omega_target if omega_target is not None else omega_d

    act_kind = params.get('act_kind', 'tanh')
    gain     = params.get('gain', 1.0)
    f, fp    = get_activation(act_kind, gain=gain)

    records = []
    total   = len(G_VALUES) * n_seeds
    idx     = 0

    for g in G_VALUES:
        for seed in range(n_seeds):
            W0 = init_W(n, g, seed=seed * 13 + 5)

            # LLE and ρ of W₀ (before any learning)
            lle_init  = compute_lle_discrete(W0, f, fp, seed=seed, **disc_kwargs)
            rho_init  = float(np.abs(np.linalg.eigvals(W0)).max())

            # Full Oja run — learning active the entire time
            t_act, x_hist, _, _, W_final = simulate_oja(
                W0, params, T=T, dt=dt,
                rho_sample_every=999999,   # skip rho tracking
                seed=seed,
            )

            # R² over the post-transient window
            r2 = compute_r2(t_act, x_hist, omega_t, transient_frac=0.3)

            # LLE and ρ of W_final (after T units of learning)
            lle_final = compute_lle_discrete(W_final, f, fp, seed=seed, **disc_kwargs)
            rho_final = float(np.abs(np.linalg.eigvals(W_final)).max())

            records.append(dict(
                g=g, seed=seed,
                lle_initial=float(lle_init),
                lle_final=float(lle_final),
                rho_initial=rho_init,
                rho_final=rho_final,
                r2=float(r2),
            ))
            idx += 1
            if verbose:
                print(f"  Dir2 [{idx:3d}/{total}]  g={g:.2f}  "
                      f"LLE_init={lle_init:+.3f}  LLE_final={lle_final:+.3f}"
                      f"  R²={r2:.3f}")

    return pd.DataFrame(records)


def plot_direction2(df, save_dir=_HERE):
    """Generate and save all Direction 2 plots."""
    summary = (
        df.groupby('g')
        .agg(
            r2_mean=('r2', 'mean'),
            r2_std=('r2', 'std'),
            lle_init_mean=('lle_initial', 'mean'),
            lle_final_mean=('lle_final', 'mean'),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Plot 1: R² vs initial g  (bar chart)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(summary))
    ax.bar(x_pos, summary['r2_mean'], yerr=summary['r2_std'],
           color='steelblue', alpha=0.75, capsize=4,
           error_kw=dict(lw=1.5, ecolor='navy'))
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{g:.2g}" for g in summary['g']], fontsize=9)
    ax.axhline(0.0, color='gray', lw=0.8, linestyle='--')
    ax.set_xlabel('initial spectral radius  g', fontsize=12)
    ax.set_ylabel('R²  (mean ± std, during Oja learning)', fontsize=12)
    ax.set_title('Fourier reconstruction quality vs initial spectral radius\n'
                 '(measured during active Oja adaptation, T=80)', fontsize=11)
    ax.set_ylim(bottom=min(-0.05, summary['r2_mean'].min() - 0.05))
    fig.tight_layout()
    p1 = os.path.join(save_dir, 'r2_vs_g.png')
    fig.savefig(p1, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p1}")

    # ------------------------------------------------------------------
    # Plot 2: R² vs initial LLE
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(
        df['lle_initial'], df['r2'],
        c=df['g'], cmap='plasma', s=40, alpha=0.75, zorder=3,
    )
    _add_binned_trend(ax, df['lle_initial'].values, df['r2'].values)
    ax.axvline(0.0, linestyle='--', color='black', lw=1.0,
               label='LLE = 0  (edge of chaos)')
    plt.colorbar(sc, ax=ax, label='initial g')
    ax.set_xlabel('initial LLE  λ₁  (autonomous map, before any learning)', fontsize=11)
    ax.set_ylabel('R²  (during Oja learning)', fontsize=12)
    ax.set_title('Is R² maximized at the edge of chaos?  (W₀ before learning)',
                 fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(save_dir, 'r2_vs_initial_lle.png')
    fig.savefig(p2, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p2}")

    # ------------------------------------------------------------------
    # Plot 3: R² vs post-learning LLE
    # ------------------------------------------------------------------
    finite_mask = np.isfinite(df['lle_final'])
    df_finite   = df[finite_mask]

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(
        df_finite['lle_final'], df_finite['r2'],
        c=df_finite['g'], cmap='plasma', s=40, alpha=0.75, zorder=3,
    )
    _add_binned_trend(ax, df_finite['lle_final'].values, df_finite['r2'].values)
    ax.axvline(0.0, linestyle='--', color='black', lw=1.0,
               label='LLE = 0  (edge of chaos)')
    plt.colorbar(sc, ax=ax, label='initial g')
    ax.set_xlabel('LLE  λ₁  after T=80 of Oja learning  (autonomous map)', fontsize=11)
    ax.set_ylabel('R²  (during Oja learning)', fontsize=12)
    ax.set_title('Is R² maximized at the edge of chaos?  (W after T=80 learning)',
                 fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p3 = os.path.join(save_dir, 'r2_vs_final_lle.png')
    fig.savefig(p3, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p3}")

    # ------------------------------------------------------------------
    # Plot 4: initial LLE vs post-learning LLE
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df_finite['lle_initial'], df_finite['lle_final'],
        c=df_finite['g'], cmap='plasma', s=40, alpha=0.75, zorder=3,
    )
    all_lle = np.concatenate([df_finite['lle_initial'].values,
                               df_finite['lle_final'].values])
    finite  = all_lle[np.isfinite(all_lle)]
    if len(finite) > 1:
        lim = [finite.min() - 0.05, finite.max() + 0.05]
        ax.plot(lim, lim, 'k--', lw=0.9, label='no change (identity)')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.axhline(0.0, color='gray', lw=0.7, linestyle=':')
    ax.axvline(0.0, color='gray', lw=0.7, linestyle=':')
    plt.colorbar(sc, ax=ax, label='initial g')
    ax.set_xlabel('initial LLE  (W₀, before any learning)', fontsize=11)
    ax.set_ylabel('post-learning LLE  (W after T=80)', fontsize=11)
    ax.set_title('How does T=80 of Oja learning shift the LLE?', fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p4 = os.path.join(save_dir, 'lle_initial_vs_final.png')
    fig.savefig(p4, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p4}")

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(save_dir, 'direction2_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def _add_binned_trend(ax, x_vals, y_vals, n_bins=8):
    """Overlay mean ± std of y in equal-count bins of x (finite values only)."""
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x, y = x_vals[mask], y_vals[mask]
    if len(x) < n_bins * 2:
        return
    order  = np.argsort(x)
    x, y   = x[order], y[order]
    splits = np.array_split(np.arange(len(x)), n_bins)
    x_mid, y_mu, y_std = [], [], []
    for idx in splits:
        if len(idx) == 0:
            continue
        x_mid.append(x[idx].mean())
        y_mu.append(y[idx].mean())
        y_std.append(y[idx].std())
    x_mid = np.array(x_mid)
    y_mu  = np.array(y_mu)
    y_std = np.array(y_std)
    ax.plot(x_mid, y_mu, color='black', lw=2.0, zorder=5, label='binned mean')
    ax.fill_between(x_mid, y_mu - y_std, y_mu + y_std,
                    color='black', alpha=0.12, zorder=4)


def _default_params():
    return {
        'tau': 1.0, 'eta': 0.01, 'lambda': 0.001,
        'gain': 1.0, 'act_kind': 'tanh',
        'A': 1.0, 'omega_drive': 1.0,
    }
