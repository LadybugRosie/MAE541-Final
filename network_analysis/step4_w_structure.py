"""
step4_w_structure.py
====================
W structural geometry vs computational quality.

Question
--------
Is rank or effective dimensionality of W the mediating variable
that connects g, LLE, and R²?

Parts
-----
A — Rank, PR, Frobenius norm, and condition number kappa tracked over
    T=400 of Oja learning for g in {0.3, 1.0, 2.0}.

B — For each (g, seed) from direction2 approach (T=80), compute
    rank(W_final) and PR(W_final), then scatter vs R².
    R² loaded from oja_chaos/direction2_data.csv when available,
    recomputed otherwise.

C — Pearson correlation heatmap: {g, lle_initial, rank_final, PR_final,
    rho_final} × R².

Outputs
-------
w_structure_over_time.png  — 4-panel time traces for 3 g values
rank_vs_r2.png             — scatter: rank(W_final) vs R², colored by g
pr_vs_r2.png               — scatter: PR(W_final) vs R², colored by g
correlation_heatmap.png    — Pearson correlation matrix heatmap
step4_data.csv             — g, seed, t, rank, PR, frob, kappa, rho, r2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'chaos_stability'))
sys.path.insert(0, os.path.join(_HERE, '..', 'oja_chaos'))

from rnn_models import init_W, get_activation
from oja_core import simulate_oja, compute_r2

G_VALUES   = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]
G_TRACE    = [0.3, 1.0, 2.0]
DIR2_CSV   = os.path.join(_HERE, '..', 'oja_chaos', 'direction2_data.csv')


def _w_structure(W):
    """Return rank, PR, frob, kappa for a weight matrix W."""
    sigmas = np.linalg.svd(W, compute_uv=False)
    rank   = int(np.linalg.matrix_rank(W, tol=1e-3))
    s2sum  = float((sigmas ** 2).sum())
    PR     = float(sigmas.sum() ** 2 / s2sum) if s2sum > 1e-12 else 0.0
    frob   = float(np.linalg.norm(W, 'fro'))
    nz     = sigmas[sigmas > 1e-10]
    kappa  = float(nz[0] / nz[-1]) if len(nz) > 1 else np.inf
    return rank, PR, frob, kappa


def _run_oja_structure_trace(W0, params, dt=0.05, sample_every=20, seed=0):
    """
    Run Oja for T=400 and track W structural metrics every `sample_every` steps.
    Returns (t_list, rank_list, PR_list, frob_list, kappa_list, rho_list).
    """
    n           = W0.shape[0]
    tau         = params.get('tau', 1.0)
    eta         = params.get('eta', 0.01)
    lam         = params.get('lambda', 0.001)
    gain        = params.get('gain', 1.0)
    act_kind    = params.get('act_kind', 'tanh')
    A           = params.get('A', 1.0)
    omega_drive = params.get('omega_drive', 1.0)

    f, _ = get_activation(act_kind, gain=gain)
    rng  = np.random.default_rng(seed)
    x    = rng.normal(0.0, 0.01, n)
    phi  = rng.uniform(0.0, 2.0 * np.pi, n)
    W    = W0.copy().astype(float)

    T_total = 400.0
    n_steps = int(round(T_total / dt))

    t_list, rank_list, PR_list, frob_list, kappa_list, rho_list = [], [], [], [], [], []

    def _rhs(x_, t_):
        drive = A * np.sin(omega_drive * t_ + phi)
        return (-x_ + f(W @ x_ + drive)) / tau

    for i in range(n_steps):
        t = i * dt

        if i % sample_every == 0:
            rank, PR, frob, kappa = _w_structure(W)
            rho = float(np.abs(np.linalg.eigvals(W)).max())
            t_list.append(t)
            rank_list.append(rank)
            PR_list.append(PR)
            frob_list.append(frob)
            kappa_list.append(kappa)
            rho_list.append(rho)

        k1 = _rhs(x, t)
        k2 = _rhs(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = _rhs(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = _rhs(x + dt * k3,        t + dt)
        x_new = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.all(np.isfinite(x_new)):
            break

        dW = eta * (np.outer(x, x) - (x ** 2)[:, None] * W) - lam * W
        W  = W + dt * dW
        x  = x_new

    return t_list, rank_list, PR_list, frob_list, kappa_list, rho_list


def run_step4(n=50, n_seeds=5, params=None, dt=0.05, verbose=True):
    """
    Part A: structural traces for G_TRACE.
    Part B: rank/PR vs R² using direction2 grid (G_VALUES x n_seeds, T=80).

    Returns DataFrame with per-point data.
    """
    if params is None:
        params = _default_params()
    omega_d = params.get('omega_drive', 1.0)

    # ---- Part A: structure over time ----
    trace_records = []
    for g in G_TRACE:
        if verbose:
            print(f"  Step4A  g={g:.2f}  T=400 structural trace")
        W0 = init_W(n, g, seed=7)
        ts, ranks, PRs, frobs, kappas, rhos = _run_oja_structure_trace(
            W0, params, dt=dt, sample_every=20, seed=0,
        )
        for t_, r_, pr_, fr_, k_, rho_ in zip(ts, ranks, PRs, frobs, kappas, rhos):
            trace_records.append(dict(
                g=g, part='A', t=t_,
                rank=r_, PR=pr_, frob=fr_, kappa=k_, rho=rho_, r2=np.nan,
            ))

    # ---- Part B: rank/PR vs R² ----
    dir2_df = None
    if os.path.exists(DIR2_CSV):
        dir2_df = pd.read_csv(DIR2_CSV)

    b_records = []
    total = len(G_VALUES) * n_seeds
    idx   = 0

    for g in G_VALUES:
        for seed in range(n_seeds):
            W0 = init_W(n, g, seed=seed * 13 + 5)

            # Get R² — prefer loading from direction2_data.csv
            if dir2_df is not None:
                row = dir2_df[(dir2_df['g'] == g) & (dir2_df['seed'] == seed)]
                if len(row) == 1:
                    r2 = float(row['r2'].values[0])
                else:
                    t_act, x_hist, _, _, _ = simulate_oja(
                        W0, params, T=80.0, dt=dt,
                        rho_sample_every=999999, seed=seed,
                    )
                    r2 = compute_r2(t_act, x_hist, omega_d, transient_frac=0.3)
            else:
                t_act, x_hist, _, _, _ = simulate_oja(
                    W0, params, T=80.0, dt=dt,
                    rho_sample_every=999999, seed=seed,
                )
                r2 = compute_r2(t_act, x_hist, omega_d, transient_frac=0.3)

            # W_final for rank/PR
            _, _, _, _, W_final = simulate_oja(
                W0, params, T=80.0, dt=dt,
                rho_sample_every=999999, seed=seed,
            )
            rank_f, PR_f, frob_f, kappa_f = _w_structure(W_final)
            rho_f = float(np.abs(np.linalg.eigvals(W_final)).max())

            b_records.append(dict(
                g=g, part='B', t=80.0,
                rank=rank_f, PR=PR_f, frob=frob_f, kappa=kappa_f,
                rho=rho_f, r2=float(r2),
            ))
            idx += 1
            if verbose:
                print(f"  Step4B [{idx:3d}/{total}]  g={g:.2f}  "
                      f"rank={rank_f}  PR={PR_f:.2f}  R²={r2:.3f}")

    df = pd.DataFrame(trace_records + b_records)
    return df


def plot_step4(df, save_dir=_HERE):
    """Save all Step 4 plots."""
    _plot_structure_over_time(df[df['part'] == 'A'], save_dir)
    df_b = df[df['part'] == 'B'].copy()
    _plot_rank_vs_r2(df_b, save_dir)
    _plot_pr_vs_r2(df_b, save_dir)
    _plot_correlation_heatmap(df_b, save_dir)

    csv_path = os.path.join(save_dir, 'step4_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def _plot_structure_over_time(df_a, save_dir):
    """4-panel: rank, PR, frob, kappa vs time for G_TRACE."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    color_map = {0.3: 'steelblue', 1.0: 'tomato', 2.0: 'forestgreen'}
    metrics   = ['rank', 'PR', 'frob', 'kappa']
    ylabels   = ['rank(W)', 'participation ratio  PR',
                 'Frobenius norm  ‖W‖_F', 'condition number  κ']
    titles    = ['Matrix rank', 'Effective dimensionality  PR',
                 'Weight magnitude  ‖W‖_F', 'Condition number  κ = σ_max/σ_min']

    for ax, metric, ylabel, title in zip(axes, metrics, ylabels, titles):
        for g in G_TRACE:
            sub = df_a[df_a['g'] == g].sort_values('t')
            if sub.empty:
                continue
            vals = sub[metric].values
            if metric == 'kappa':
                finite = np.isfinite(vals)
                ax.semilogy(sub['t'].values[finite], vals[finite],
                            'o-', color=color_map[g], lw=2.0, markersize=3,
                            label=f'g={g}')
            else:
                ax.plot(sub['t'], vals, 'o-', color=color_map[g],
                        lw=2.0, markersize=3, label=f'g={g}')
        ax.set_xlabel('time', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle('W structural geometry during Oja learning (T=400)', fontsize=12)
    fig.tight_layout()
    p = os.path.join(save_dir, 'w_structure_over_time.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_rank_vs_r2(df_b, save_dir):
    """Scatter: rank(W_final) vs R², colored by g."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(df_b['rank'], df_b['r2'],
                    c=df_b['g'], cmap='plasma', s=40, alpha=0.75, zorder=3)
    plt.colorbar(sc, ax=ax, label='initial g')
    ax.set_xlabel('rank(W)  after T=80 of Oja learning', fontsize=12)
    ax.set_ylabel('R²  (Fourier reconstruction)', fontsize=12)
    ax.set_title('Does matrix rank predict reconstruction quality?', fontsize=11)
    fig.tight_layout()
    p = os.path.join(save_dir, 'rank_vs_r2.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_pr_vs_r2(df_b, save_dir):
    """Scatter: PR(W_final) vs R², colored by g."""
    sys.path.insert(0, os.path.join(_HERE, '..', 'oja_chaos'))
    try:
        from direction2 import _add_binned_trend
        has_binned = True
    except ImportError:
        has_binned = False

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(df_b['PR'], df_b['r2'],
                    c=df_b['g'], cmap='plasma', s=40, alpha=0.75, zorder=3)
    if has_binned:
        _add_binned_trend(ax, df_b['PR'].values, df_b['r2'].values)
    plt.colorbar(sc, ax=ax, label='initial g')
    ax.set_xlabel('participation ratio  PR(W)  after T=80', fontsize=12)
    ax.set_ylabel('R²  (Fourier reconstruction)', fontsize=12)
    ax.set_title('Effective dimensionality PR vs reconstruction quality', fontsize=11)
    fig.tight_layout()
    p = os.path.join(save_dir, 'pr_vs_r2.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_correlation_heatmap(df_b, save_dir):
    """Pearson correlation heatmap for structural predictors of R²."""
    cols = ['g', 'rho', 'rank', 'PR', 'frob', 'r2']
    labels = ['g', 'ρ(W)', 'rank(W)', 'PR(W)', '‖W‖_F', 'R²']
    sub = df_b[cols].dropna()
    if sub.empty or len(sub) < 4:
        print("  (skipping heatmap — insufficient data)")
        return

    C = sub.corr(method='pearson').values
    n = len(cols)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C, vmin=-1.0, vmax=1.0, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{C[i, j]:.2f}", ha='center', va='center',
                    fontsize=9, color='black' if abs(C[i, j]) < 0.7 else 'white')
    ax.set_title('Pearson correlations: structural predictors of R²', fontsize=12)
    fig.tight_layout()
    p = os.path.join(save_dir, 'correlation_heatmap.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _default_params():
    return {
        'tau': 1.0, 'eta': 0.01, 'lambda': 0.001,
        'gain': 1.0, 'act_kind': 'tanh',
        'A': 1.0, 'omega_drive': 1.0,
    }
