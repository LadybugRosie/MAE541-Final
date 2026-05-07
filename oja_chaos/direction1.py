"""
direction1.py
=============
Direction 1: Oja plasticity as a self-organizing edge-of-chaos regulator.

Question
--------
Does Oja learning drive ρ(W) toward g ≈ 1 (the edge of chaos) on its own,
regardless of where the network starts?

Method
------
Initialize networks at g₀ ∈ {0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5}, run the
Oja-driven simulation, and track ρ(W) = max|eigenvalue(W)| over time.
Average across n_seeds random initial conditions.

Outputs
-------
spectral_radius_trajectories.png
spectral_radius_final.png
direction1_data.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'chaos_stability'))
sys.path.insert(0, _HERE)

from rnn_models import init_W
from oja_core import simulate_oja

G_INITIAL = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5]


def _sweep_rho(n, n_seeds, params, T, dt, rho_sample_every, label, verbose):
    """Run ρ(W) sweep for all g₀ values under the given params."""
    results = {}
    for g0 in G_INITIAL:
        rho_over_seeds = []
        for seed in range(n_seeds):
            W0 = init_W(n, g0, seed=seed * 7 + 3)
            _, _, rho_hist, t_rho, _ = simulate_oja(
                W0, params, T=T, dt=dt,
                rho_sample_every=rho_sample_every, seed=seed,
            )
            rho_over_seeds.append(rho_hist)
            if verbose:
                print(f"  Dir1 [{label}]  g₀={g0:.2f}  seed={seed}"
                      f"  ρ_init={rho_hist[0]:.3f}  ρ_final={rho_hist[-1]:.3f}")
        rho_arr = np.array(rho_over_seeds)
        results[g0] = {
            't':        t_rho,
            'rho_mean': rho_arr.mean(axis=0),
            'rho_std':  rho_arr.std(axis=0),
            'rho_all':  rho_arr,
        }
    return results


def run_direction1(
    n=50, n_seeds=5,
    params=None, T=400.0, dt=0.05,
    rho_sample_every=20, verbose=True,
):
    """
    Sweep initial g₀ values; track ρ(W) over time for two conditions:
      - Driven   (A = 1.0): sinusoidal input present during learning
      - Autonomous (A = 0.0): no external drive, purely recurrent

    Returns
    -------
    (results_driven, results_auto) — each a dict keyed by g₀ with:
        't'        : (K,)           sample time points
        'rho_mean' : (K,)           mean ρ(W) across seeds
        'rho_std'  : (K,)           std ρ(W) across seeds
        'rho_all'  : (n_seeds, K)   raw ρ(W) per seed
    """
    if params is None:
        params = _default_params()

    p_driven = dict(params)
    p_auto   = dict(params, A=0.0)

    print("  [driven]")
    res_driven = _sweep_rho(n, n_seeds, p_driven, T, dt, rho_sample_every,
                            'driven', verbose)
    print("  [autonomous]")
    res_auto   = _sweep_rho(n, n_seeds, p_auto, T, dt, rho_sample_every,
                            'auto', verbose)
    return res_driven, res_auto


def plot_direction1(res_driven, res_auto, save_dir=_HERE):
    """
    Generate and save all Direction 1 plots.
    """
    g_vals = sorted(res_driven.keys())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(g_vals)))

    # ------------------------------------------------------------------
    # Plot 1: ρ(W) trajectories — driven vs autonomous, side by side
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                              sharey=False, sharex=False)
    for ax, results, title in zip(
        axes,
        [res_driven, res_auto],
        ['Driven  (A = 1,  sinusoidal input)',
         'Autonomous  (A = 0,  no input)'],
    ):
        for g0, color in zip(g_vals, colors):
            r   = results[g0]
            mu  = r['rho_mean']
            std = r['rho_std']
            ax.plot(r['t'], mu, color=color, lw=1.8, label=f"g₀ = {g0}")
            ax.fill_between(r['t'], mu - std, mu + std,
                            color=color, alpha=0.13)
        ax.axhline(1.0, linestyle='--', color='black', lw=1.0,
                   label='ρ = 1  (edge)')
        ax.set_xlabel('time', fontsize=12)
        ax.set_ylabel('spectral radius  ρ(W)', fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle('Oja learning: does ρ(W) self-organize toward the edge of chaos?',
                 fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(save_dir, 'spectral_radius_trajectories.png')
    fig.savefig(p1, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p1}")

    # ------------------------------------------------------------------
    # Plot 2: Final ρ(W) driven vs autonomous (paired bar chart)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    x_pos  = np.arange(len(g_vals))
    width  = 0.35
    driven_finals = [res_driven[g]['rho_all'][:, -1].mean() for g in g_vals]
    auto_finals   = [res_auto[g]['rho_all'][:, -1].mean()   for g in g_vals]
    driven_stds   = [res_driven[g]['rho_all'][:, -1].std()  for g in g_vals]
    auto_stds     = [res_auto[g]['rho_all'][:, -1].std()    for g in g_vals]

    ax.bar(x_pos - width / 2, driven_finals, width, yerr=driven_stds,
           label='Driven (A=1)', color='tomato', alpha=0.75, capsize=4)
    ax.bar(x_pos + width / 2, auto_finals,   width, yerr=auto_stds,
           label='Autonomous (A=0)', color='steelblue', alpha=0.75, capsize=4)
    ax.axhline(1.0, linestyle='--', color='black', lw=1.0,
               label='ρ = 1  (edge of chaos)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"g₀={g:.1f}" for g in g_vals], fontsize=9)
    ax.set_xlabel('initial spectral radius g₀', fontsize=12)
    ax.set_ylabel('final ρ(W) after learning', fontsize=12)
    ax.set_title('Final spectral radius: driven vs autonomous Oja learning', fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(save_dir, 'spectral_radius_final.png')
    fig.savefig(p2, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p2}")

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    records = []
    for condition, results in [('driven', res_driven), ('autonomous', res_auto)]:
        for g0 in g_vals:
            r = results[g0]
            for s in range(r['rho_all'].shape[0]):
                for k, (t, rho) in enumerate(zip(r['t'], r['rho_all'][s])):
                    records.append(dict(condition=condition, g0=g0,
                                        seed=s, t=t, rho=rho))
    df = pd.DataFrame(records)
    csv_path = os.path.join(save_dir, 'direction1_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def _default_params():
    return {
        'tau': 1.0, 'eta': 0.01, 'lambda': 0.001,
        'gain': 1.0, 'act_kind': 'tanh',
        'A': 1.0, 'omega_drive': 1.0,
    }
