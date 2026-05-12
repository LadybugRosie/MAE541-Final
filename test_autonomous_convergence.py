"""
test_autonomous_convergence.py
===============================
Three tests for whether the autonomous Oja case converges to g=1
on a longer timescale.

Test A — Long-time extension (T=5000, autonomous only)
Test B — Phase portrait dρ/dt vs ρ(W) from existing CSV
Test C — Analytical check: slow-manifold fixed point for sub/super-critical A=0

Run:
    uv run python test_autonomous_convergence.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'oja_chaos'))
sys.path.insert(0, os.path.join(ROOT, 'chaos_stability'))

from rnn_models import init_W
from oja_core import simulate_oja

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   9,
    'legend.framealpha': 0.85,
    'lines.linewidth':   2.0,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})
SAVEKW = dict(dpi=150, bbox_inches='tight')

PLASMA   = plt.cm.plasma
G_VALS   = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5]
COLORS   = {g: PLASMA(i / (len(G_VALS) - 1))
            for i, g in enumerate(G_VALS)}

PARAMS_AUTO = {
    'tau': 1.0, 'eta': 0.01, 'lambda': 0.001,
    'gain': 1.0, 'act_kind': 'tanh',
    'A': 0.0, 'omega_drive': 1.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Test B  (immediate — uses existing direction1_data.csv)
# ─────────────────────────────────────────────────────────────────────────────
def test_B():
    print("\n=== Test B: Phase portrait dρ/dt vs ρ(W) from existing CSV ===")
    df = pd.read_csv(os.path.join(ROOT, 'oja_chaos', 'direction1_data.csv'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cond, title in zip(
        axes,
        ['driven', 'autonomous'],
        ['Sinusoidal Drive (A = 1)', 'Autonomous Network (A = 0)'],
    ):
        sub = df[df['condition'] == cond].sort_values(['g0', 'seed', 't'])
        for g0 in G_VALS:
            gs = sub[sub['g0'] == g0]
            if gs.empty:
                continue
            # average rho over seeds at each time point
            mean_rho = gs.groupby('t')['rho'].mean()
            t_vals   = mean_rho.index.values
            rho_vals = mean_rho.values

            if len(rho_vals) < 2:
                continue
            # finite-difference dρ/dt
            drho_dt = np.diff(rho_vals) / np.diff(t_vals)
            rho_mid  = 0.5 * (rho_vals[:-1] + rho_vals[1:])

            ax.scatter(rho_mid, drho_dt,
                       color=COLORS[g0], s=6, alpha=0.5, label=f'$g_0 = {g0}$')

        ax.axhline(0.0, color='black', lw=1.0, linestyle='--', label='$d\\rho/dt = 0$')
        ax.axvline(1.0, color='gray',  lw=0.9, linestyle=':', label='$\\rho = 1$')
        ax.set_xlabel('Spectral radius  $\\rho(W)$')
        ax.set_ylabel('$d\\rho/dt$  (steps$^{-1}$)')
        ax.set_title(title)
        # one legend entry per g0 (deduplicate)
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(list(seen.values()), list(seen.keys()), fontsize=8, markerscale=3)

    fig.suptitle('Phase Portrait: $d\\rho/dt$ vs $\\rho(W)$\n'
                 'Nullcline at $d\\rho/dt = 0$ shows whether dynamics attract toward $\\rho = 1$',
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    p = os.path.join(ROOT, 'test_B_phase_portrait.png')
    fig.savefig(p, **SAVEKW)
    plt.close(fig)
    print(f"  Saved: {p}")

    # --- print summary statistics ---
    for cond in ['driven', 'autonomous']:
        sub = df[df['condition'] == cond]
        print(f"\n  [{cond}] Final ρ(W) at T=400 per g0 (mean ± std across seeds):")
        for g0 in G_VALS:
            gs = sub[sub['g0'] == g0]
            if gs.empty:
                continue
            final_rho = gs.groupby('seed')['rho'].last()
            print(f"    g0={g0:.2f}  →  ρ_final = {final_rho.mean():.3f} ± {final_rho.std():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test A  (long-run simulation, autonomous only, T=5000)
# ─────────────────────────────────────────────────────────────────────────────
def test_A(n=50, n_seeds=2, T=5000.0, dt=0.1, rho_sample_every=200):
    print(f"\n=== Test A: Long-run autonomous (T={T}, dt={dt}) ===")
    results = {}
    t0 = time.time()
    total = len(G_VALS) * n_seeds
    done  = 0
    for g0 in G_VALS:
        rho_over_seeds = []
        t_rho = None
        for seed in range(n_seeds):
            W0 = init_W(n, g0, seed=seed * 7 + 3)
            _, _, rho_hist, t_r, _ = simulate_oja(
                W0, PARAMS_AUTO, T=T, dt=dt,
                rho_sample_every=rho_sample_every, seed=seed,
            )
            rho_over_seeds.append(rho_hist)
            t_rho = t_r
            done += 1
            elapsed = time.time() - t0
            remaining = elapsed / done * (total - done)
            print(f"  [A] {done}/{total}  g0={g0:.2f} seed={seed}"
                  f"  ρ_final={rho_hist[-1]:.3f}"
                  f"  (~{remaining:.0f}s left)")
        rho_arr = np.array(rho_over_seeds)
        results[g0] = {
            't':        t_rho,
            'rho_mean': rho_arr.mean(axis=0),
            'rho_std':  rho_arr.std(axis=0),
        }

    # --- plot ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for g0 in G_VALS:
        r   = results[g0]
        mu  = r['rho_mean']
        std = r['rho_std']
        ax.plot(r['t'], mu, color=COLORS[g0], lw=1.8, label=f'$g_0 = {g0}$')
        ax.fill_between(r['t'], mu - std, mu + std, color=COLORS[g0], alpha=0.12)

    ax.axhline(1.0, linestyle='--', color='black', lw=1.0,
               label='$\\rho = 1$ (stability boundary)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Spectral radius  $\\rho(W)$')
    ax.set_title(f'Autonomous Oja Dynamics Over Long Run ($T = {int(T)}$)\n'
                 '\\small{Does $\\rho(W)$ converge to 1 at longer timescales?}')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(ROOT, 'test_A_long_autonomous.png')
    fig.savefig(p, **SAVEKW)
    plt.close(fig)
    print(f"  Saved: {p}")

    # --- summary table ---
    print(f"\n  [A] Final ρ(W) at T={T} (mean across {n_seeds} seeds):")
    print(f"  {'g0':>6}  {'ρ(T=400)':>10}  {'ρ(T='+str(int(T))+')':>12}  {'drift':>10}")
    df_old = pd.read_csv(os.path.join(ROOT, 'oja_chaos', 'direction1_data.csv'))
    for g0 in G_VALS:
        old_rho = df_old[(df_old['condition']=='autonomous') & (df_old['g0']==g0)
                        ].groupby('seed')['rho'].last().mean()
        new_rho = results[g0]['rho_mean'][-1]
        drift   = new_rho - old_rho
        print(f"  {g0:>6.2f}  {old_rho:>10.3f}  {new_rho:>12.3f}  {drift:>+10.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test C  (analytical verification with quick numerical check)
# ─────────────────────────────────────────────────────────────────────────────
def test_C():
    print("\n=== Test C: Analytical slow-manifold fixed point (A=0) ===")
    print("""
  Theory:
  -------
  The Oja weight dynamics average to:
      dW/dt ≈ η [C(W) - C_ii(W) W] - λ W
  where C(W) = (1/T_avg) ∫ x(t;W) x(t;W)^T dt.

  CASE 1: g < 1 (sub-critical, autonomous)
    The origin x*=0 is globally attracting (Banach contraction since ρ(W)<1).
    With x≈0:  C(W) ≈ 0
    Slow-manifold:  dW/dt ≈ -λ W
    Fixed point:    W* = 0  →  ρ(W*) = 0.
    Prediction: ρ(W) → 0 exponentially at rate λ = 0.001
                Timescale: 1/λ = 1000 time units.

  CASE 2: g > 1 (super-critical, autonomous)
    Network is chaotic; activity has isotropic covariance C ≈ σ²I.
    Slow-manifold fixed point: W* = (η σ²) / (η σ² + λ) · I  (diagonal)
    ρ(W*) = (η σ²) / (η σ² + λ).
    For σ² ≈ 0.5 (tanh output variance under chaos), η=0.01, λ=0.001:
    ρ(W*) ≈ (0.01·0.5) / (0.01·0.5 + 0.001) = 0.005/0.006 ≈ 0.83.
    Prediction: ρ(W) → ~0.83 (NOT 1) for all super-critical initial conditions.

  CONCLUSION:
    Neither sub-critical nor super-critical autonomous networks converge to g=1.
    Sub-critical → ρ→0 (decays to silence).
    Super-critical → ρ→~σ²η/(σ²η+λ) < 1 (shrinks below 1, bounded by activity variance).
    The driven case converges to g≈45 because sinusoidal drive forces C to rank-1
    with large norm, creating a self-reinforcing feedback; white noise creates C≈σ²I
    like the autonomous chaotic case.
""")

    # Numerical verification: short run to check direction
    print("  Numerical check (T=2000, sub/super-critical):")
    n = 50
    params_auto = dict(PARAMS_AUTO)

    for g0, label in [(0.5, 'sub-critical'), (2.0, 'super-critical')]:
        rho_seq = []
        for seed in range(3):
            W0 = init_W(n, g0, seed=seed * 7 + 3)
            _, _, rho_hist, t_r, _ = simulate_oja(
                W0, params_auto, T=2000.0, dt=0.1,
                rho_sample_every=500, seed=seed,
            )
            rho_seq.append(rho_hist)
        rho_arr = np.array(rho_seq)
        t_checkpoints = t_r[::1]   # all sample points
        print(f"    g0={g0} ({label}):")
        for ti, (tm, ts) in zip(t_r, zip(rho_arr.mean(0), rho_arr.std(0))):
            print(f"      t={ti:6.0f}  ρ = {tm:.4f} ± {ts:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_B()
    test_A()
    test_C()
    print("\n=== All tests complete ===")
