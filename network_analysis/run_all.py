"""
run_all.py
==========
CLI entry point for the network_analysis four-step pipeline.

Usage
-----
    # Full run (all steps):
    uv run python network_analysis/run_all.py

    # Quick smoke test (2 seeds, skip avalanche):
    uv run python network_analysis/run_all.py --seeds 2 --quick

    # Skip individual steps:
    uv run python network_analysis/run_all.py --skip step3

    # Adjust network size or time:
    uv run python network_analysis/run_all.py --n 50 --seeds 5
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")   # headless — must precede any pyplot import

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from step1_baseline      import run_step1, plot_step1
from step2_spectrum      import run_step2, plot_step2
from step3_phase_transition import run_step3, plot_step3
from step4_w_structure   import run_step4, plot_step4


def main():
    parser = argparse.ArgumentParser(
        description="Network analysis: 4-step pipeline"
    )
    parser.add_argument('--n',      type=int,   default=50,
                        help='network size (default 50)')
    parser.add_argument('--seeds',  type=int,   default=5,
                        help='random seeds per data point (default 5)')
    parser.add_argument('--dt',     type=float, default=0.05,
                        help='integration step size (default 0.05)')
    parser.add_argument('--eta',    type=float, default=0.01,
                        help='Oja learning rate eta (default 0.01)')
    parser.add_argument('--lam',    type=float, default=0.001,
                        help='weight decay lambda (default 0.001)')
    parser.add_argument('--quick',  action='store_true',
                        help='skip slow avalanche computation (step3b)')
    parser.add_argument('--skip',   nargs='*',  default=[],
                        help='steps to skip: step1 step2 step3 step4')
    args = parser.parse_args()

    params = {
        'tau':        1.0,
        'eta':        args.eta,
        'lambda':     args.lam,
        'gain':       1.0,
        'act_kind':   'tanh',
        'A':          1.0,
        'omega_drive': 1.0,
    }

    sep = '=' * 70

    # ------------------------------------------------------------------
    # Step 1: Frozen baseline vs Oja-learned comparison
    # ------------------------------------------------------------------
    if 'step1' not in args.skip:
        print(f'\n{sep}')
        print('Step 1: Baseline — frozen reservoir vs Oja-learned R²')
        print(sep)
        df1 = run_step1(
            n=args.n, n_seeds=args.seeds,
            params=params, T=80.0, dt=args.dt,
            verbose=True,
        )
        plot_step1(df1, save_dir=_HERE)

    # ------------------------------------------------------------------
    # Step 2: Eigenvalue spectrum evolution
    # ------------------------------------------------------------------
    if 'step2' not in args.skip:
        print(f'\n{sep}')
        print('Step 2: Eigenvalue spectrum evolution during Oja learning')
        print(sep)
        df2, eig_data = run_step2(
            n=args.n, n_seeds=1,   # 1 seed: qualitative spectrum visualisation
            params=params, dt=args.dt,
            verbose=True,
        )
        plot_step2(df2, eig_data, save_dir=_HERE)

    # ------------------------------------------------------------------
    # Step 3: Phase transition physics
    # ------------------------------------------------------------------
    if 'step3' not in args.skip:
        print(f'\n{sep}')
        print('Step 3: Phase transition — critical slowing + avalanches')
        print(sep)
        df3, f_act = run_step3(
            n=args.n, n_seeds=args.seeds,
            params=params, dt=args.dt,
            skip_avalanche=args.quick,
            verbose=True,
        )
        plot_step3(df3, f_act, n=args.n, params=params, save_dir=_HERE)

    # ------------------------------------------------------------------
    # Step 4: W structure vs computation
    # ------------------------------------------------------------------
    if 'step4' not in args.skip:
        print(f'\n{sep}')
        print('Step 4: W structural geometry vs reconstruction quality')
        print(sep)
        df4 = run_step4(
            n=args.n, n_seeds=args.seeds,
            params=params, dt=args.dt,
            verbose=True,
        )
        plot_step4(df4, save_dir=_HERE)

    print(f'\nAll outputs saved to:  {_HERE}')


if __name__ == '__main__':
    main()
