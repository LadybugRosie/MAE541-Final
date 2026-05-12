"""
run_geff.py
===========
CLI entry point for the effective gain g_eff experiment.

Usage
-----
    # Full run (2 seeds, T=300):
    uv run python geff_analysis/run_geff.py

    # Quick smoke test (1 seed, T=150):
    uv run python geff_analysis/run_geff.py --seeds 1 --T 150

    # Custom learning params:
    uv run python geff_analysis/run_geff.py --eta 0.02 --lam 0.002
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from geff_core import run_geff_experiment, run_static_scan, plot_geff


def main():
    parser = argparse.ArgumentParser(description='Effective gain g_eff experiment')
    parser.add_argument('--n',     type=int,   default=50,    help='network size (default 50)')
    parser.add_argument('--seeds', type=int,   default=2,     help='random seeds (default 2)')
    parser.add_argument('--T',     type=float, default=300.0, help='simulation duration (default 300)')
    parser.add_argument('--dt',    type=float, default=0.05,  help='integration step size (default 0.05)')
    parser.add_argument('--eta',   type=float, default=0.01,  help='Oja learning rate η (default 0.01)')
    parser.add_argument('--lam',   type=float, default=0.001, help='weight decay λ (default 0.001)')
    args = parser.parse_args()

    params = {
        'tau': 1.0, 'eta': args.eta, 'lambda': args.lam,
        'gain': 1.0, 'act_kind': 'tanh',
        'A': 1.0, 'omega_drive': 1.0,
    }

    print('=' * 70)
    print('Effective gain g_eff experiment')
    print(f'n={args.n}  seeds={args.seeds}  T={args.T}  dt={args.dt}')
    print(f'η={args.eta}  λ={args.lam}')
    print('=' * 70)

    print('\n[A+B] g_eff dynamics during Oja learning (sin and white drives)...')
    trace_df = run_geff_experiment(
        n=args.n, n_seeds=args.seeds,
        params=params, T=args.T, dt=args.dt,
    )

    print('\n[C] Static g_eff scan (frozen weights, sinusoidal drive)...')
    scan_df = run_static_scan(
        n=args.n, n_seeds=args.seeds,
        params=params, T=args.T, dt=args.dt,
    )

    print('\nGenerating plots...')
    plot_geff(trace_df, scan_df, save_dir=_HERE)

    print(f'\nAll outputs saved to:  {_HERE}')


if __name__ == '__main__':
    main()
