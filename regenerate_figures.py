"""
regenerate_figures.py
=====================
Loads cached CSVs and regenerates all 6 paper-ready figures.

Run:
    uv run python regenerate_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# ── paper-style rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.titlesize':     11,
    'axes.titleweight':   'normal',
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '0.8',
    'lines.linewidth':    2.0,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'grid.color':         '0.85',
    'grid.linewidth':     0.6,
})

SAVEKW = dict(dpi=150, bbox_inches='tight')

ROOT     = os.path.dirname(os.path.abspath(__file__))
OJA_DIR  = os.path.join(ROOT, 'oja_chaos')
NET_DIR  = os.path.join(ROOT, 'network_analysis')
SIG_DIR  = os.path.join(ROOT, 'signal_comparison')
GEFF_DIR = os.path.join(ROOT, 'geff_analysis')

sys.path.insert(0, os.path.join(ROOT, 'chaos_stability'))
from rnn_models import init_W, init_h0, get_activation, run_discrete, run_continuous

# ── colour palettes ───────────────────────────────────────────────────────────
PLASMA = plt.cm.plasma

G_VALS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5]
G_COLORS = {g: PLASMA(i / (len(G_VALS) - 1)) for i, g in enumerate(G_VALS)}

_DRIVE_COLORS = {
    'sin':   '#d62728',
    'quasi': '#ff7f0e',
    'white': '#1f77b4',
    'zero':  '#7f7f7f',
}
_DRIVE_LABELS = {
    'sin':   r'Sinusoidal  ($\mathrm{rank}(C) = 1$)',
    'quasi': r'Quasi-periodic  ($\mathrm{rank}(C) = 3$)',
    'white': r'White noise  ($\mathrm{rank}(C) \approx n$)',
    'zero':  'No input (autonomous)',
}


def _add_panel_label(ax, letter, x=-0.12, y=1.05):
    """Add bold panel label (A), (B), etc."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='left')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — spectral_radius_trajectories.png
# ─────────────────────────────────────────────────────────────────────────────
def fig_spectral_radius():
    df = pd.read_csv(os.path.join(OJA_DIR, 'direction1_data.csv'))

    # Condition names in CSV: 'driven' and 'autonomous'
    conditions = [('driven', 'Sinusoidal Drive ($A = 1$)'),
                  ('autonomous', 'Autonomous Network ($A = 0$)')]

    g_vals  = sorted(df['g0'].unique())
    n_g     = len(g_vals)
    colors  = [PLASMA(i / (n_g - 1)) for i in range(n_g)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (cond, subtitle) in zip(axes, conditions):
        sub = df[df['condition'] == cond]
        for g0, color in zip(g_vals, colors):
            s   = sub[sub['g0'] == g0]
            if s.empty:
                continue
            mu  = s.groupby('t')['rho'].mean()
            std = s.groupby('t')['rho'].std().fillna(0)
            ax.plot(mu.index, mu.values, color=color, lw=1.8,
                    label=f'$g_0 = {g0}$')
            ax.fill_between(mu.index, mu - std, mu + std,
                            color=color, alpha=0.12)

        ax.axhline(1.0, linestyle='--', color='black', lw=1.1,
                   label=r'$\rho = 1$')
        ax.set_xlabel('Time')
        ax.set_ylabel(r'Spectral radius $\rho(W)$')
        ax.set_title(subtitle, pad=8)
        ax.legend(fontsize=8, ncol=2, loc='upper left')

    _add_panel_label(axes[0], 'A')
    _add_panel_label(axes[1], 'B')

    fig.suptitle('Spectral Radius Under Oja Plasticity: Driven vs. Autonomous',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    p = os.path.join(OJA_DIR, 'spectral_radius_trajectories.png')
    fig.savefig(p, **SAVEKW)
    plt.close(fig)
    print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — r2_vs_g.png
# ─────────────────────────────────────────────────────────────────────────────
def fig_r2_vs_g():
    df = pd.read_csv(os.path.join(OJA_DIR, 'direction2_data.csv'))
    summary = (df.groupby('g')
                 .agg(r2_mean=('r2', 'mean'), r2_std=('r2', 'std'))
                 .reset_index())

    peak_idx = summary['r2_mean'].idxmax()
    peak_g   = summary.loc[peak_idx, 'g']

    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(summary))

    bars = ax.bar(x_pos, summary['r2_mean'],
                  color='steelblue', alpha=0.80, zorder=3)
    ax.errorbar(x_pos, summary['r2_mean'], yerr=summary['r2_std'],
                fmt='none', ecolor='navy', elinewidth=1.5, capsize=4, zorder=4)

    # Highlight peak bar
    bars[peak_idx].set_color('#1a5a8a')
    bars[peak_idx].set_edgecolor('black')
    bars[peak_idx].set_linewidth(1.2)

    ax.axhline(0.0, color='gray', lw=0.8, linestyle='--')
    ax.axvline(peak_idx, color='tomato', linestyle=':', lw=1.5, alpha=0.8,
               label=f'Peak at $g_0 = {peak_g}$')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{g:.2g}' for g in summary['g']])
    ax.set_xlabel(r'Initial spectral radius $g_0$')
    ax.set_ylabel(r'Reconstruction $R^2$ (mean $\pm$ std)')
    ax.set_title(
        'Signal Reconstruction Quality vs. Initial Spectral Radius\n'
        r'Linear readout during Oja adaptation, $T = 80$, sinusoidal drive',
        pad=8, fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    p = os.path.join(OJA_DIR, 'r2_vs_g.png')
    fig.savefig(p, **SAVEKW)
    plt.close(fig)
    print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — r2_baseline_vs_oja.png
# ─────────────────────────────────────────────────────────────────────────────
def fig_r2_baseline():
    df = pd.read_csv(os.path.join(NET_DIR, 'step1_data.csv'))
    summary = (df.groupby(['g', 'condition'])
                 .agg(r2_mean=('r2', 'mean'), r2_std=('r2', 'std'))
                 .reset_index())
    frozen = summary[summary['condition'] == 'frozen'].sort_values('g')
    oja    = summary[summary['condition'] == 'oja'].sort_values('g')
    g_vals = frozen['g'].values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(g_vals, frozen['r2_mean'], yerr=frozen['r2_std'],
                fmt='o-', color='steelblue', capsize=4, lw=2.0,
                label=r'Frozen reservoir (ridge, $\alpha = 10^{-3}$)')
    ax.errorbar(g_vals, oja['r2_mean'], yerr=oja['r2_std'],
                fmt='s--', color='tomato', capsize=4, lw=2.0,
                label='Oja-learned weights (OLS)')
    ax.axhline(0.0, color='gray', lw=0.8, linestyle='--')
    ax.set_xlabel(r'Initial spectral radius $g_0$')
    ax.set_ylabel(r'Reconstruction $R^2$ (mean $\pm$ std)')
    ax.set_title(
        'Frozen Random Reservoir vs. Oja-Learned Weights\n'
        r'$T = 80$, sinusoidal drive, $n = 50$ neurons',
        pad=8, fontsize=11,
    )
    ax.legend(loc='center right')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    p = os.path.join(NET_DIR, 'r2_baseline_vs_oja.png')
    fig.savefig(p, **SAVEKW)
    plt.close(fig)
    print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — correlation_rank_by_drive.png  (2×2, all four drives)
# ─────────────────────────────────────────────────────────────────────────────
def fig_correlation_rank():
    df = pd.read_csv(os.path.join(SIG_DIR, 'comparison_data.csv'))
    g_target    = 1.0
    drive_order = ['sin', 'quasi', 'white', 'zero']
    panel_labels = ['A', 'B', 'C', 'D']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()

    for ax, drive_kind, panel in zip(axes_flat, drive_order, panel_labels):
        color = _DRIVE_COLORS[drive_kind]
        sub   = df[(df['drive'] == drive_kind) & (df['g'] == g_target)]
        if sub.empty:
            ax.set_title(f'{_DRIVE_LABELS[drive_kind]}  (no data)')
            continue

        mean = sub.groupby('t')['rank_C'].mean()
        std  = sub.groupby('t')['rank_C'].std().fillna(0)

        ax.plot(mean.index, mean.values, color=color, lw=2.2)
        ax.fill_between(mean.index,
                        (mean - std).clip(lower=0), mean + std,
                        color=color, alpha=0.18)
        ax.axhline(1.0,  linestyle='--', color='black', lw=1.0,
                   label='rank $= 1$')
        ax.axhline(50.0, linestyle=':',  color='0.5',   lw=0.9,
                   label='rank $= n = 50$')
        ax.set_title(_DRIVE_LABELS[drive_kind], pad=6)
        ax.set_xlabel('Time')
        ax.set_ylabel(r'rank$(\langle xx^T \rangle)$')
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0, top=55)
        _add_panel_label(ax, panel)

    fig.suptitle(
        r'Rank of Activity Correlation Matrix $\langle xx^T \rangle$ by Drive Type  ($g = 1.0$)'
        '\nRolling 200-step window; rank collapse to 1 indicates rank-deficient correlations',
        fontsize=12, fontweight='bold', y=1.02,
    )
    fig.tight_layout()
    p = os.path.join(SIG_DIR, 'correlation_rank_by_drive.png')
    fig.savefig(p, **SAVEKW)
    plt.close(fig)
    print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — geff_static_scan.png
# ─────────────────────────────────────────────────────────────────────────────
def fig_geff_scan():
    df = pd.read_csv(os.path.join(GEFF_DIR, 'geff_scan.csv'))
    summary = (df.groupby('g')
                 .agg(g_eff_mean=('g_eff',  'mean'), g_eff_std=('g_eff',  'std'),
                      rho_mean  =('rho_DW', 'mean'), rho_std  =('rho_DW', 'std'))
                 .reset_index())

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(summary['g'], summary['g'], 'k:', lw=1.0, alpha=0.45,
            label=r'$g_\mathrm{eff} = g$ (no saturation)')
    ax.errorbar(summary['g'], summary['rho_mean'],
                yerr=summary['rho_std'].fillna(0),
                fmt='s--', color='tomato', capsize=4, lw=2.0,
                label=r'$\rho(D(t)W)$ (direct eigenvalue)')
    ax.errorbar(summary['g'], summary['g_eff_mean'],
                yerr=summary['g_eff_std'].fillna(0),
                fmt='o-', color='steelblue', capsize=4, lw=2.0,
                label=r'$g_\mathrm{eff}$ (heuristic)')

    ax.axhline(1.0, color='0.4', lw=0.9, linestyle=':')
    ax.axvline(1.0, color='0.4', lw=0.9, linestyle='-.', alpha=0.6)

    ax.set_xlabel(r'Nominal spectral radius $g$')
    ax.set_ylabel('Effective gain')
    ax.set_title(
        'Effective Gain Under Sinusoidal Drive: Static Scan (Frozen Weights)\n'
        r'$g_\mathrm{eff} = 1$ crossing shifts from $g = 1$ to $g \approx 1.5$ due to neural saturation',
        pad=8, fontsize=11,
    )
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    p = os.path.join(GEFF_DIR, 'geff_static_scan.png')
    fig.savefig(p, **SAVEKW)
    plt.close(fig)
    print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — critical_slowing_down.png
# ─────────────────────────────────────────────────────────────────────────────
def _perturbed_divergence(W, f, n, tau=1.0, T_warmup=300, T_meas=80,
                           dt=0.05, eps=1e-6, seed=0):
    h0 = init_h0(n, seed=seed)
    _, H_warm = run_continuous(h0, W, f, tau, T_warmup, dt=0.1)
    if H_warm is None or not np.all(np.isfinite(H_warm[-1])):
        return None, None
    h0a = H_warm[-1]
    d0  = np.zeros(n); d0[0] = eps
    t1, H1 = run_continuous(h0a,       W, f, tau, T_meas, dt=dt)
    t2, H2 = run_continuous(h0a + d0,  W, f, tau, T_meas, dt=dt)
    if H1 is None or H2 is None:
        return None, None
    if not np.all(np.isfinite(H1)) or not np.all(np.isfinite(H2)):
        return None, None
    return t1, np.linalg.norm(H1 - H2, axis=1)


def fig_critical_slowing():
    df = pd.read_csv(os.path.join(NET_DIR, 'step3_data.csv'))
    n  = 50
    f, _ = get_activation('tanh', gain=1.0)

    summary = (df.groupby('g')
                 .agg(tau_mean=('tau_relax', 'mean'),
                      tau_std =('tau_relax', 'std'),
                      lle_mean=('lle', 'mean'))
                 .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: tau_relax vs g (sub-critical only) ──────────────────────────
    ax = axes[0]
    sub_crit = summary[summary['tau_mean'].notna()]
    ax.errorbar(sub_crit['g'], sub_crit['tau_mean'],
                yerr=sub_crit['tau_std'].fillna(0),
                fmt='o-', color='steelblue', capsize=4, lw=2.0)
    ax.axvline(1.0, linestyle='--', color='black', lw=1.1,
               label='$g = 1$ (critical point)')
    ax.set_xlabel(r'Spectral radius $g$')
    ax.set_ylabel(r'Relaxation time $\tau_\mathrm{relax}$ (steps)')
    ax.set_title(r'Relaxation Time Diverges as $g \to 1^-$', pad=8)
    ax.legend(fontsize=9)
    _add_panel_label(ax, 'A')

    # ── Right: analytic exponential using measured LLE from step3_data ───
    ax = axes[1]
    G_PLOT  = [0.5, 1.0, 2.0]
    colors  = ['steelblue', 'black', 'tomato']
    labels  = ['$g = 0.5$ (stable,  $\\lambda = -0.68$)',
               '$g = 1.0$ (critical, $\\lambda \\approx 0$)',
               '$g = 2.0$ (chaotic,  $\\lambda = +0.10$)']
    EPS    = 1e-6
    T_MAX  = 100
    t_plot = np.linspace(0, T_MAX, 500)
    lle_by_g = summary.set_index('g')['lle_mean']

    for g_plot, color, label in zip(G_PLOT, colors, labels):
        lle   = lle_by_g.loc[g_plot]
        delta = EPS * np.exp(lle * t_plot)
        delta = np.clip(delta, 1e-14, 20)
        ax.semilogy(t_plot, delta, color=color, lw=2.2, label=label)

    ax.axhline(EPS, color='0.5', lw=0.9, linestyle=':',
               label=r'$\varepsilon = 10^{-6}$ (initial perturbation)')
    ax.set_ylim(1e-14, 1e-1)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\|\delta h(t)\|$ (log scale)')
    ax.set_title(r'Predicted Perturbation Growth: $\|\delta h(t)\| \approx \varepsilon\,e^{\lambda t}$'
                 '\nLyapunov exponents from power-iteration measurement',
                 pad=8)
    ax.legend(fontsize=9)
    _add_panel_label(ax, 'B')

    fig.suptitle(r'Critical Slowing Down at the Edge-of-Chaos Phase Transition ($g = 1$)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    p = os.path.join(NET_DIR, 'critical_slowing_down.png')
    fig.savefig(p, **SAVEKW)
    plt.close(fig)
    print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Regenerating paper-ready figures ...')
    fig_spectral_radius()
    fig_r2_vs_g()
    fig_r2_baseline()
    fig_correlation_rank()
    fig_geff_scan()
    fig_critical_slowing()
    print('Done.')
