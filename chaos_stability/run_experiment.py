"""
run_experiment.py
=================
Entry point for the chaos / stability experiment.

Usage
-----
    # Full run (all four sweeps, ~20 min on a laptop):
    uv run python chaos_stability/run_experiment.py

    # Quick smoke-test (skip slow sweeps B and C):
    uv run python chaos_stability/run_experiment.py --quick

Outputs are saved inside chaos_stability/.
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")   # headless — must precede pyplot import

# ensure imports resolve when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from sweep import sweep_A, sweep_B, sweep_C, sweep_D
from visualize import (
    plot_lle_vs_g,
    plot_phase_diagram,
    plot_trajectory_divergence,
    plot_autocorrelation,
    plot_psd,
    plot_n_dependence,
    plot_tau_sweep,
    plot_sample_trajectories,
)

_HERE = os.path.dirname(__file__)


def _p(fname):
    """Resolve output path relative to this file's directory."""
    return os.path.join(_HERE, fname)


def run_chaos_experiment(
    n=50,
    act_kind="tanh",
    gain=1.0,
    tau=1.0,
    n_seeds=5,
    skip_sweeps=None,
    verbose=True,
):
    """
    Full chaos / stability experiment pipeline.

    Parameters
    ----------
    n           : default network size for sweeps A, B, D
    act_kind    : default activation function for sweeps A, C, D
    gain        : activation gain (applied inside activation function)
    tau         : time constant for continuous-flow ODE
    n_seeds     : random W realisations averaged per data point
    skip_sweeps : list of sweep letters to skip, e.g. ['B', 'C']
    verbose     : print progress

    Returns
    -------
    (df_A, df_B, df_C, df_D)  — DataFrames, None if sweep was skipped
    """
    skip_sweeps = set(skip_sweeps or [])

    # LLE computation kwargs — tune here to trade speed vs accuracy
    disc_kwargs = dict(n_warmup=200, n_steps=1000)
    flow_kwargs = dict(T_warmup=20.0, T_measure=50.0, dt=0.05, n_reorth=100)

    sep = "=" * 70

    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("Chaos / Stability Experiment — Map vs. Flow")
    print(sep)

    # ------------------------------------------------------------------
    print("\n[0] Sample trajectories")
    plot_sample_trajectories(
        n=n, act_kind=act_kind, gain=gain, tau=tau,
        save_path=_p("sample_trajectories.png"),
    )

    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("[A] LLE vs spectral radius g")
    print(sep)
    df_A = sweep_A(n=n, act_kind=act_kind, gain=gain, tau=tau,
                   n_seeds=n_seeds, disc_kwargs=disc_kwargs,
                   flow_kwargs=flow_kwargs, verbose=verbose)
    df_A.to_csv(_p("sweep_A.csv"), index=False)
    plot_lle_vs_g(df_A, save_path=_p("lle_vs_g.png"))

    # ------------------------------------------------------------------
    if "B" not in skip_sweeps:
        print(f"\n{sep}")
        print("[B] Activation function comparison")
        print(sep)
        df_B = sweep_B(n=n, tau=tau, gain=gain, n_seeds=n_seeds,
                       disc_kwargs=disc_kwargs, flow_kwargs=flow_kwargs,
                       verbose=verbose)
        df_B.to_csv(_p("sweep_B.csv"), index=False)
        plot_phase_diagram(df_B, save_path=_p("phase_diagram.png"))
    else:
        print("\n[B] Skipped.")
        df_B = None

    # ------------------------------------------------------------------
    if "C" not in skip_sweeps:
        print(f"\n{sep}")
        print("[C] Network-size dependence")
        print(sep)
        df_C = sweep_C(act_kind=act_kind, gain=gain, tau=tau,
                       n_seeds=n_seeds, disc_kwargs=disc_kwargs,
                       flow_kwargs=flow_kwargs, verbose=verbose)
        df_C.to_csv(_p("sweep_C.csv"), index=False)
        plot_n_dependence(df_C, save_path=_p("n_dependence.png"))
    else:
        print("\n[C] Skipped.")
        df_C = None

    # ------------------------------------------------------------------
    if "D" not in skip_sweeps:
        print(f"\n{sep}")
        print("[D] Time-constant τ sweep (continuous flow)")
        print(sep)
        df_D = sweep_D(n=n, act_kind=act_kind, gain=gain, n_seeds=n_seeds,
                       flow_kwargs=flow_kwargs, verbose=verbose)
        df_D.to_csv(_p("sweep_D.csv"), index=False)
        plot_tau_sweep(df_D, save_path=_p("tau_sweep.png"))
    else:
        print("\n[D] Skipped.")
        df_D = None

    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("[Q] Qualitative plots (divergence, autocorrelation, PSD)")
    print(sep)
    plot_trajectory_divergence(
        n=n, act_kind=act_kind, gain=gain, tau=tau,
        save_path=_p("trajectory_divergence.png"),
    )
    plot_autocorrelation(
        n=n, act_kind=act_kind, gain=gain, tau=tau,
        save_path=_p("autocorrelation.png"),
    )
    plot_psd(
        n=n, act_kind=act_kind, gain=gain, tau=tau,
        save_path=_p("power_spectrum.png"),
    )

    print(f"\nAll outputs saved to:  {_HERE}")
    return df_A, df_B, df_C, df_D


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chaos/stability experiment: map vs. flow RNNs"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Skip sweeps B and C (fast smoke-test)")
    parser.add_argument("--n",        type=int,   default=50)
    parser.add_argument("--act",      type=str,   default="tanh",
                        help="Default activation function")
    parser.add_argument("--gain",     type=float, default=1.0)
    parser.add_argument("--tau",      type=float, default=1.0)
    parser.add_argument("--seeds",    type=int,   default=5,
                        help="Random seeds per data point")
    args = parser.parse_args()

    skip = ["B", "C"] if args.quick else []

    run_chaos_experiment(
        n=args.n,
        act_kind=args.act,
        gain=args.gain,
        tau=args.tau,
        n_seeds=args.seeds,
        skip_sweeps=skip,
        verbose=True,
    )
