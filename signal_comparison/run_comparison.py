"""
run_comparison.py
=================
CLI entry point for the signal structure vs rank collapse experiment.

Usage
-----
    # Full run (3 seeds, T=300):
    uv run python signal_comparison/run_comparison.py

    # Quick smoke test (2 seeds, T=150):
    uv run python signal_comparison/run_comparison.py --seeds 2 --T 150

    # Adjust learning params:
    uv run python signal_comparison/run_comparison.py --eta 0.02 --lam 0.002
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")   # headless — must precede any pyplot import

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from compare_signals import run_comparison, plot_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Signal structure vs Oja rank collapse"
    )
    parser.add_argument('--n',     type=int,   default=50,
                        help='network size (default 50)')
    parser.add_argument('--seeds', type=int,   default=3,
                        help='random seeds per condition (default 3)')
    parser.add_argument('--T',     type=float, default=300.0,
                        help='long-trace simulation duration (default 300)')
    parser.add_argument('--dt',    type=float, default=0.05,
                        help='integration step size (default 0.05)')
    parser.add_argument('--eta',   type=float, default=0.01,
                        help='Oja learning rate η (default 0.01)')
    parser.add_argument('--lam',   type=float, default=0.001,
                        help='weight decay λ (default 0.001)')
    args = parser.parse_args()

    params = {
        'tau':         1.0,
        'eta':         args.eta,
        'lambda':      args.lam,
        'gain':        1.0,
        'act_kind':    'tanh',
        'A':           1.0,
        'omega_drive': 1.0,
    }

    print('=' * 70)
    print('Signal structure vs Oja rank collapse')
    print(f'n={args.n}  seeds={args.seeds}  T={args.T}  dt={args.dt}')
    print(f'η={args.eta}  λ={args.lam}')
    print('=' * 70)

    trace_df, r2_df = run_comparison(
        n=args.n, n_seeds=args.seeds,
        params=params, T=args.T, dt=args.dt,
        T_r2=80.0, verbose=True,
    )
    plot_comparison(trace_df, r2_df, save_dir=_HERE)
    print(f'\nAll outputs saved to:  {_HERE}')


if __name__ == '__main__':
    main()
