"""
step2_spectrum.py
=================
Eigenvalue spectrum evolution during Oja learning.

Question
--------
How does Oja learning reshape W's eigenvalue distribution?
Does it collapse from a Girko disk to a low-rank outlier structure?

Method
------
For g in {0.3, 1.0, 2.0}, run a continuous Oja simulation for T=400.
Capture W at snapshot times t in {0, 10, 30, 80, 200, 400}:
  - Full complex eigenvalue spectrum (n=50 eigenvalues)
  - Singular value spectrum
  - rank(W), participation ratio PR = (sum sigma)^2 / (sum sigma^2), Frobenius norm

Implemented as a single uninterrupted Oja loop so x(t) is continuous.

Outputs
-------
eigenvalue_evolution.png     — 3x6 grid (g x snapshot): eigenvalue cloud in complex plane
rank_over_time.png           — rank(W) and PR vs time for 3 g values
singular_value_spectrum.png  — sorted singular values (log scale) at each snapshot, g=1.0
step2_spectrum_data.csv      — g, seed, t, rho, rank, PR, frob, sigma_max
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

G_SNAPSHOT = [0.3, 1.0, 2.0]
SNAPSHOT_TIMES = [0, 10, 30, 80, 200, 400]


def _w_metrics(W):
    """Compute structural metrics for a weight matrix W."""
    evals  = np.linalg.eigvals(W)
    sigmas = np.linalg.svd(W, compute_uv=False)
    rank   = int(np.linalg.matrix_rank(W, tol=1e-3))
    s2sum  = float((sigmas ** 2).sum())
    PR     = float(sigmas.sum() ** 2 / s2sum) if s2sum > 0 else 0.0
    frob   = float(np.linalg.norm(W, 'fro'))
    rho    = float(np.abs(evals).max())
    return dict(evals=evals, sigmas=sigmas, rank=rank, PR=PR, frob=frob, rho=rho)


def _run_oja_snapshots(W0, params, snapshot_times, dt=0.05, seed=0):
    """
    Run Oja simulation for max(snapshot_times) time units.
    Capture full W metrics at each snapshot time.

    Returns dict: {t_snap: metrics_dict}
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

    T_total = float(max(snapshot_times))
    n_steps = int(round(T_total / dt))

    # Map step index → snapshot time
    snap_at = {}
    for ts in snapshot_times:
        step_idx = int(round(ts / dt))
        snap_at[step_idx] = ts

    snapshots = {}
    if 0 in snap_at:
        snapshots[0] = _w_metrics(W)

    def _rhs(x_, t_):
        drive = A * np.sin(omega_drive * t_ + phi)
        return (-x_ + f(W @ x_ + drive)) / tau

    for i in range(n_steps):
        t = i * dt
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

        next_step = i + 1
        if next_step in snap_at:
            ts = snap_at[next_step]
            snapshots[ts] = _w_metrics(W)

    return snapshots


def run_step2(n=50, n_seeds=1, params=None, dt=0.05, verbose=True):
    """
    Eigenvalue spectrum evolution for G_SNAPSHOT g values.

    Returns (DataFrame of structural metrics, nested dict of eigenvalue arrays)
    """
    if params is None:
        params = _default_params()

    records  = []
    eig_data = {}   # {(g, seed): {t: metrics_dict}}

    for g in G_SNAPSHOT:
        for seed in range(n_seeds):
            if verbose:
                print(f"  Step2  g={g:.2f}  seed={seed}  T=400")
            W0    = init_W(n, g, seed=seed * 13 + 5)
            snaps = _run_oja_snapshots(W0, params, SNAPSHOT_TIMES, dt=dt, seed=seed)
            eig_data[(g, seed)] = snaps

            for ts, m in sorted(snaps.items()):
                records.append(dict(
                    g=g, seed=seed, t=ts,
                    rho=m['rho'], rank=m['rank'], PR=m['PR'], frob=m['frob'],
                    sigma_max=float(m['sigmas'][0]) if len(m['sigmas']) > 0 else np.nan,
                ))

    return pd.DataFrame(records), eig_data


def plot_step2(df, eig_data, save_dir=_HERE):
    """Save all Direction 2 / Step 2 plots."""
    _plot_eigenvalue_grid(eig_data, save_dir)
    _plot_rank_over_time(df, save_dir)
    _plot_singular_value_spectrum(eig_data, save_dir)

    csv_path = os.path.join(save_dir, 'step2_spectrum_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def _plot_eigenvalue_grid(eig_data, save_dir):
    """3×6 grid: eigenvalue cloud in complex plane at each (g, snapshot_time)."""
    n_g   = len(G_SNAPSHOT)
    n_t   = len(SNAPSHOT_TIMES)
    fig, axes = plt.subplots(n_g, n_t, figsize=(3.2 * n_t, 3.0 * n_g),
                              sharex=False, sharey=False)

    for row, g in enumerate(G_SNAPSHOT):
        # Use seed=0 for display
        snaps = eig_data.get((g, 0), {})
        for col, ts in enumerate(SNAPSHOT_TIMES):
            ax  = axes[row, col]
            m   = snaps.get(ts)
            if m is None:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
                continue
            evals = m['evals']
            ax.scatter(evals.real, evals.imag, s=15, alpha=0.7, color='steelblue')
            # Theoretical disk radius
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(f't = {ts}', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'g = {g}', fontsize=9)
            ax.axhline(0, color='gray', lw=0.5, linestyle=':')
            ax.axvline(0, color='gray', lw=0.5, linestyle=':')
            ax.tick_params(labelsize=7)
            rho = m['rho']
            ax.set_title(f't={ts}  ρ={rho:.1f}', fontsize=8)

    fig.suptitle('Eigenvalue spectrum of W during Oja learning\n'
                 '(rows: g=0.3/1.0/2.0 | cols: snapshot times)',
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(save_dir, 'eigenvalue_evolution.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_rank_over_time(df, save_dir):
    """rank(W) and PR vs time for 3 g values."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'0.3': 'steelblue', '1.0': 'tomato', '2.0': 'forestgreen'}
    color_list = ['steelblue', 'tomato', 'forestgreen']

    for i, g in enumerate(G_SNAPSHOT):
        sub = df[df['g'] == g].sort_values('t')
        if sub.empty:
            continue
        col = color_list[i]
        axes[0].plot(sub['t'], sub['rank'], 'o-', color=col, lw=2.0, label=f'g={g}')
        axes[1].plot(sub['t'], sub['PR'],   'o-', color=col, lw=2.0, label=f'g={g}')

    axes[0].set_xlabel('time', fontsize=12)
    axes[0].set_ylabel('rank(W)', fontsize=12)
    axes[0].set_title('Matrix rank during Oja learning', fontsize=11)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel('time', fontsize=12)
    axes[1].set_ylabel('participation ratio  PR', fontsize=12)
    axes[1].set_title('Effective dimensionality  PR = (Σσ)² / Σσ²', fontsize=11)
    axes[1].legend(fontsize=9)

    fig.suptitle('Structural collapse of W during Oja learning', fontsize=12)
    fig.tight_layout()
    p = os.path.join(save_dir, 'rank_over_time.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_singular_value_spectrum(eig_data, save_dir):
    """Sorted singular values (log scale) at each snapshot for g=1.0."""
    g_target = 1.0
    snaps = eig_data.get((g_target, 0), {})
    if not snaps:
        return

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(SNAPSHOT_TIMES)))
    fig, ax = plt.subplots(figsize=(9, 5))

    for ts, color in zip(SNAPSHOT_TIMES, colors):
        m = snaps.get(ts)
        if m is None:
            continue
        sigmas = m['sigmas']
        ax.semilogy(np.arange(1, len(sigmas) + 1), np.sort(sigmas)[::-1],
                    'o-', color=color, lw=1.5, markersize=4, label=f't={ts}')

    ax.set_xlabel('singular value rank', fontsize=12)
    ax.set_ylabel('singular value  σ  (log scale)', fontsize=12)
    ax.set_title(f'Singular value spectrum evolution during Oja learning  (g={g_target})',
                 fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    p = os.path.join(save_dir, 'singular_value_spectrum.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def _default_params():
    return {
        'tau': 1.0, 'eta': 0.01, 'lambda': 0.001,
        'gain': 1.0, 'act_kind': 'tanh',
        'A': 1.0, 'omega_drive': 1.0,
    }
