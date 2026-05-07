"""
run_experiments.py
==================
Entry point for both Oja-chaos directions.

Usage
-----
    # Full run (both directions):
    uv run python oja_chaos/run_experiments.py

    # Skip one direction:
    uv run python oja_chaos/run_experiments.py --skip direction2

    # Faster smoke-test (fewer seeds, shorter T):
    uv run python oja_chaos/run_experiments.py --seeds 2 --T 100

    # Adjust learning parameters:
    uv run python oja_chaos/run_experiments.py --eta 0.02 --lam 0.002
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")   # headless — must precede pyplot import

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from direction1 import run_direction1, plot_direction1
from direction2 import run_direction2, plot_direction2


def main():
    parser = argparse.ArgumentParser(
        description="Oja plasticity × chaos: directions 1 and 2"
    )
    parser.add_argument('--n',     type=int,   default=50,
                        help='network size (default 50)')
    parser.add_argument('--seeds', type=int,   default=5,
                        help='random seeds per data point (default 5)')
    parser.add_argument('--T',     type=float, default=400.0,
                        help='simulation duration (default 400)')
    parser.add_argument('--dt',    type=float, default=0.05,
                        help='integration step size (default 0.05)')
    parser.add_argument('--eta',   type=float, default=0.01,
                        help='Oja learning rate η (default 0.01)')
    parser.add_argument('--lam',   type=float, default=0.001,
                        help='weight decay λ (default 0.001)')
    parser.add_argument('--gain',  type=float, default=1.0,
                        help='activation gain (default 1.0)')
    parser.add_argument('--act',   type=str,   default='tanh',
                        help='activation function (default tanh)')
    parser.add_argument('--skip',  nargs='*',  default=[],
                        help='direction1 and/or direction2')
    args = parser.parse_args()

    params = {
        'tau':        1.0,
        'eta':        args.eta,
        'lambda':     args.lam,
        'gain':       args.gain,
        'act_kind':   args.act,
        'A':          1.0,
        'omega_drive': 1.0,
    }

    sep = '=' * 70

    if 'direction1' not in args.skip:
        print(f'\n{sep}')
        print('Direction 1: Oja spectral self-organization')
        print(sep)
        res_driven, res_auto = run_direction1(
            n=args.n, n_seeds=args.seeds,
            params=params, T=args.T, dt=args.dt,
            verbose=True,
        )
        plot_direction1(res_driven, res_auto, save_dir=_HERE)

    if 'direction2' not in args.skip:
        print(f'\n{sep}')
        print('Direction 2: R² vs LLE — edge-of-chaos Fourier reconstruction')
        print(sep)
        # T=80 is the sweet spot: transients gone, Oja not yet saturated.
        df = run_direction2(
            n=args.n, n_seeds=args.seeds,
            params=params, T=80.0, dt=args.dt,
            verbose=True,
        )
        plot_direction2(df, save_dir=_HERE)

    print(f'\nAll outputs saved to:  {_HERE}')


if __name__ == '__main__':
    main()
