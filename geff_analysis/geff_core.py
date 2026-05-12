"""
geff_core.py
============
Tests the effective random-matrix gain heuristic g_eff(t) from the slide
"Driven Dynamics: Effective Random-Matrix Gain."

Theory
------
Linearized perturbation dynamics near driven trajectory x*(t):
  τ·δẋ = [-I + D(t)W] δx,   D(t) = diag(sech²(h_i(t))),   h_i = [Wx + b]_i

Heuristic spectral radius of D(t)W:
  g_eff(t) ≈ (‖W(t)‖_F / √n) · √(mean_i d_i(t)²)
Derivation: Var((DW)_ij) = d_i² · Var(W_ij);
  g_eff² = n · mean(Var((DW)_ij)) = [‖W‖_F²/n] · mean(d_i²)

Three experiments
-----------------
A.  Oja dynamics (sin vs white drive): does g_eff stay bounded as ρ(W)→45?
    (Prediction: yes — saturation drives d_i→0 faster than ‖W‖_F grows.)
B.  Heuristic accuracy: scatter g_eff vs ρ(D(t)W(t)) directly measured.
    (Prediction: points near the identity line.)
C.  Static scan (frozen weights, sin drive): is g_eff=1 a better chaos
    boundary than g=1?
    (Prediction: g_eff=1 crossing occurs at g < 1 because saturation reduces
    the effective gain even for the initial random matrix.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'chaos_stability'))

from rnn_models import init_W

G_VALUES = [0.3, 0.7, 1.0, 1.5, 2.0]
G_SCAN   = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]

_G_COLORS = {
    0.3: '#4393c3',
    0.7: '#92c5de',
    1.0: '#404040',
    1.5: '#f4a582',
    2.0: '#d6604d',
}


# ============================================================
# Drive helpers
# ============================================================

def _sin_drive(A, omega, n, seed=0):
    phi = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, n)
    return lambda t, rng: A * np.sin(omega * t + phi)


def _white_drive(A, n):
    return lambda t, rng: rng.normal(0.0, A / np.sqrt(2), n)


# ============================================================
# Core simulation
# ============================================================

def simulate_geff(W0, params, drive_fn, T=300.0, dt=0.05,
                  sample_every=20, rho_dw_every=100,
                  seed=0, frozen=False):
    """
    RK4 on x, Euler on W (η=0 if frozen=True).

    At every `sample_every` steps records:
      t, rho_W, frob_W, g_eff, mean_d2, rho_DW (NaN except at rho_dw_every steps)

    Parameters
    ----------
    frozen : bool
        If True η=0 — weights stay at W0. Used for Experiment C.

    Returns
    -------
    list of dicts
    """
    n   = W0.shape[0]
    tau = params.get('tau', 1.0)
    eta = params.get('eta', 0.01)
    lam = params.get('lambda', 0.001)

    rng = np.random.default_rng(seed)
    x   = rng.normal(0.0, 0.01, n)
    W   = W0.copy().astype(float)

    n_steps = int(round(T / dt))
    records = []

    for i in range(n_steps):
        t = i * dt
        b = drive_fn(t, rng)          # drive vector frozen across RK4 sub-stages

        # ----- g_eff metrics from current (x, W, b) -----
        h       = W @ x + b
        d       = 1.0 - np.tanh(h) ** 2    # sech²(h_i) = tanh'(h_i)
        mean_d2 = float(np.mean(d ** 2))
        frob_W  = float(np.linalg.norm(W, 'fro'))
        g_eff   = (frob_W / np.sqrt(n)) * np.sqrt(mean_d2)

        if i % sample_every == 0:
            rho_W  = float(np.abs(np.linalg.eigvals(W)).max())
            rho_DW = np.nan
            if i % rho_dw_every == 0:
                DW     = d[:, None] * W      # diag(d) @ W via O(n²) broadcast
                rho_DW = float(np.abs(np.linalg.eigvals(DW)).max())
            records.append(dict(
                t=t, rho_W=rho_W, frob_W=frob_W,
                g_eff=g_eff, mean_d2=mean_d2, rho_DW=rho_DW,
            ))

        # ----- RK4 for x (W is constant during this step) -----
        def _f(x_):
            return (-x_ + np.tanh(W @ x_ + b)) / tau

        k1 = _f(x)
        k2 = _f(x + 0.5 * dt * k1)
        k3 = _f(x + 0.5 * dt * k2)
        k4 = _f(x + dt * k3)
        x_new = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        if not np.all(np.isfinite(x_new)):
            break

        if not frozen:
            W = W + dt * (eta * (np.outer(x, x) - (x**2)[:, None] * W) - lam * W)
        x = x_new

    return records


# ============================================================
# Experiment A+B: Oja dynamics
# ============================================================

def run_geff_experiment(n=50, n_seeds=2, params=None, T=300.0, dt=0.05, verbose=True):
    """
    For each (drive∈{sin,white}, g∈G_VALUES, seed), run Oja simulation and
    record g_eff(t) alongside ρ(W)(t).

    Returns trace_df.
    """
    if params is None:
        params = _default_params()
    A     = params.get('A', 1.0)
    omega = params.get('omega_drive', 1.0)

    all_records = []
    combos = [
        (dk, g, s)
        for dk in ('sin', 'white')
        for g in G_VALUES
        for s in range(n_seeds)
    ]

    for k, (drive_kind, g, seed) in enumerate(combos):
        W0  = init_W(n, g, seed=seed * 13 + 5)
        dfn = (_sin_drive(A, omega, n, seed) if drive_kind == 'sin'
               else _white_drive(A, n))
        recs = simulate_geff(W0, params, dfn, T=T, dt=dt, seed=seed,
                              sample_every=20, rho_dw_every=100)
        for r in recs:
            all_records.append(dict(drive=drive_kind, g=g, seed=seed, **r))
        if verbose and recs:
            print(f'  [{k+1:3d}/{len(combos)}]  drive={drive_kind:5s}  g={g:.1f}  '
                  f'ρ(W)={recs[-1]["rho_W"]:.2f}  g_eff={recs[-1]["g_eff"]:.3f}')

    return pd.DataFrame(all_records)


# ============================================================
# Experiment C: static g_eff scan
# ============================================================

def run_static_scan(n=50, n_seeds=2, params=None, T=300.0, dt=0.05,
                    warmup_frac=0.7, verbose=True):
    """
    Frozen weights (η=0), sinusoidal drive across g∈G_SCAN.
    Averages g_eff and ρ(DW) over the post-warmup window.

    Returns scan_df.
    """
    if params is None:
        params = _default_params()
    A     = params.get('A', 1.0)
    omega = params.get('omega_drive', 1.0)

    scan_records = []
    combos = [(g, s) for g in G_SCAN for s in range(n_seeds)]

    for k, (g, seed) in enumerate(combos):
        W0  = init_W(n, g, seed=seed * 13 + 5)
        dfn = _sin_drive(A, omega, n, seed)
        recs = simulate_geff(W0, params, dfn, T=T, dt=dt, seed=seed,
                             frozen=True,
                             sample_every=20, rho_dw_every=20)
        if not recs:
            continue

        df_run   = pd.DataFrame(recs)
        t_cut    = df_run['t'].max() * warmup_frac
        ss       = df_run[df_run['t'] >= t_cut]

        g_eff_m  = float(ss['g_eff'].mean())
        rho_dw_m = (float(ss['rho_DW'].dropna().mean())
                    if ss['rho_DW'].notna().any() else np.nan)
        d2_m     = float(ss['mean_d2'].mean())

        scan_records.append(dict(g=g, seed=seed,
                                 g_eff=g_eff_m, rho_DW=rho_dw_m, mean_d2=d2_m))
        if verbose:
            print(f'  [{k+1:3d}/{len(combos)}]  g={g:.2f}  '
                  f'g_eff={g_eff_m:.3f}  ρ(DW)={rho_dw_m:.3f}')

    return pd.DataFrame(scan_records)


# ============================================================
# Plotting
# ============================================================

def plot_geff(trace_df, scan_df, save_dir=_HERE):
    _plot_trajectories(trace_df, save_dir)
    _plot_saturation(trace_df, save_dir)
    _plot_heuristic_vs_direct(trace_df, save_dir)
    _plot_static_scan(scan_df, save_dir)

    trace_df.to_csv(os.path.join(save_dir, 'geff_data.csv'), index=False)
    scan_df.to_csv(os.path.join(save_dir, 'geff_scan.csv'), index=False)
    print(f'  Saved CSVs to {save_dir}')


def _plot_trajectories(df, save_dir):
    """2×2: ρ(W) and g_eff vs time for sin and white drives."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    metrics = [
        ('rho_W', r'$\rho(W)$', 'Nominal spectral radius  ρ(W)'),
        ('g_eff', r'$g_{\rm eff}(t)$', 'Effective gain  g_eff(t)'),
    ]

    for col, drive_kind in enumerate(('sin', 'white')):
        sub = df[df['drive'] == drive_kind]
        drive_label = 'Sinusoidal drive' if drive_kind == 'sin' else 'White noise drive'

        for row, (metric, ylabel, title_sfx) in enumerate(metrics):
            ax = axes[row, col]
            for g in G_VALUES:
                s = sub[sub['g'] == g]
                if s.empty:
                    continue
                mean = s.groupby('t')[metric].mean()
                std  = s.groupby('t')[metric].std().fillna(0)
                c    = _G_COLORS.get(g, 'black')
                ax.plot(mean.index, mean.values, color=c, lw=2.0, label=f'g={g}')
                ax.fill_between(mean.index, mean-std, mean+std,
                                color=c, alpha=0.15)
            ax.axhline(1.0, linestyle='--', color='gray', lw=0.8, alpha=0.7)
            ax.set_title(f'{drive_label}: {title_sfx}', fontsize=10)
            ax.set_xlabel('time', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(fontsize=8)

    fig.suptitle('Nominal ρ(W) vs effective gain g_eff(t) during Oja learning\n'
                 'Does saturation self-regulate g_eff as ρ(W) → 45?', fontsize=12)
    fig.tight_layout()
    p = os.path.join(save_dir, 'geff_trajectories.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def _plot_saturation(df, save_dir):
    """2-panel: mean(sech⁴(h_i)) = mean_d2 over time, sin vs white."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, drive_kind in zip(axes, ('sin', 'white')):
        sub = df[df['drive'] == drive_kind]
        for g in G_VALUES:
            s = sub[sub['g'] == g]
            if s.empty:
                continue
            mean = s.groupby('t')['mean_d2'].mean()
            std  = s.groupby('t')['mean_d2'].std().fillna(0)
            c    = _G_COLORS.get(g, 'black')
            ax.plot(mean.index, mean.values, color=c, lw=2.0, label=f'g={g}')
            ax.fill_between(mean.index, mean-std, mean+std, color=c, alpha=0.15)
        drive_label = 'Sinusoidal' if drive_kind == 'sin' else 'White noise'
        ax.set_title(f'{drive_label}: neuron saturation over time', fontsize=11)
        ax.set_xlabel('time', fontsize=11)
        ax.set_ylabel(r'mean$(d_i^2)$ = mean$(\mathrm{sech}^4(h_i))$', fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle(r'Saturation dynamics: mean$(\mathrm{sech}^4(h_i))$ over time'
                 '\nDecays toward 0 → neurons saturate → g_eff self-regulates', fontsize=11)
    fig.tight_layout()
    p = os.path.join(save_dir, 'geff_saturation.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def _plot_heuristic_vs_direct(df, save_dir):
    """Scatter: g_eff heuristic vs ρ(DW) direct measurement."""
    valid = df[df['rho_DW'].notna()].copy()
    if valid.empty:
        print('  [skip] no rho_DW data for scatter')
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    style = {'sin': ('tomato', 'o'), 'white': ('steelblue', 's')}

    for drive_kind in ('sin', 'white'):
        sub = valid[valid['drive'] == drive_kind]
        if sub.empty:
            continue
        c, m = style[drive_kind]
        ax.scatter(sub['g_eff'], sub['rho_DW'],
                   c=c, marker=m, alpha=0.35, s=18, label=f'{drive_kind} drive')

    lim = max(valid['g_eff'].max(), valid['rho_DW'].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1.2, label='y = x (perfect)')
    ax.set_xlabel(r'Heuristic  $g_{\rm eff} = \|W\|_F / \sqrt{n} \cdot \sqrt{\mathrm{mean}(d_i^2)}$',
                  fontsize=10)
    ax.set_ylabel(r'Direct  $\rho(D(t)W(t))$', fontsize=11)
    ax.set_title('Heuristic vs direct effective gain\n'
                 'Points near y = x → approximation is accurate', fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(save_dir, 'geff_vs_rho_dw.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def _plot_static_scan(scan_df, save_dir):
    """g_eff(g) and ρ_DW(g) for frozen weights across G_SCAN."""
    if scan_df.empty:
        print('  [skip] empty scan_df')
        return

    summary = (
        scan_df.groupby('g')
        .agg(
            g_eff_mean=('g_eff', 'mean'), g_eff_std=('g_eff', 'std'),
            rho_DW_mean=('rho_DW', 'mean'), rho_DW_std=('rho_DW', 'std'),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(summary['g'], summary['g_eff_mean'], yerr=summary['g_eff_std'].fillna(0),
                fmt='o-', color='steelblue', capsize=4, lw=2.0,
                label=r'$g_{\rm eff}$ (heuristic)')
    ax.errorbar(summary['g'], summary['rho_DW_mean'], yerr=summary['rho_DW_std'].fillna(0),
                fmt='s--', color='tomato', capsize=4, lw=2.0,
                label=r'$\rho(DW)$ (direct)')
    ax.plot(summary['g'], summary['g'], 'k:', lw=1.0, alpha=0.5,
            label='g_eff = g  (no saturation)')
    ax.axhline(1.0, color='gray', lw=0.9, linestyle=':', label='y = 1 (stability boundary)')
    ax.axvline(1.0, color='gray', lw=0.9, linestyle='-.', alpha=0.5, label='g = 1 (Girko)')

    ax.set_xlabel('nominal spectral radius  g', fontsize=12)
    ax.set_ylabel('effective gain', fontsize=12)
    ax.set_title('Effective gain vs nominal g  (frozen weights, sinusoidal drive)\n'
                 'Saturation shifts the g_eff = 1 crossing relative to g = 1', fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(save_dir, 'geff_static_scan.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def _default_params():
    return {
        'tau': 1.0, 'eta': 0.01, 'lambda': 0.001,
        'gain': 1.0, 'act_kind': 'tanh',
        'A': 1.0, 'omega_drive': 1.0,
    }
