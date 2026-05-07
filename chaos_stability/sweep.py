"""
sweep.py
========
Four hyperparameter sweeps, each returning a pandas DataFrame.

Sweep A — LLE vs spectral radius g  (both formulations, one activation)
Sweep B — LLE vs g  for all 9 activation functions
Sweep C — LLE vs g  for different network sizes n
Sweep D — LLE vs g  for different time constants τ  (continuous flow only)
"""

import numpy as np
import pandas as pd

from rnn_models import init_W, get_activation, ALL_ACTIVATIONS
from chaos_metrics import compute_lle_discrete, compute_lle_continuous


# ---- canonical g grid ----
G_VALUES = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0,
            1.05, 1.1, 1.2, 1.5, 2.0, 3.0]


# ---- shared LLE computation for one (g, params) point ----

def _lle_point(g, n, act_kind, gain, tau, n_seeds,
               disc_kwargs, flow_kwargs):
    f, fp = get_activation(act_kind, gain=gain)
    lles_map, lles_flow = [], []
    for seed in range(n_seeds):
        W = init_W(n, g, seed=seed * 31 + 7)
        lle_m = compute_lle_discrete(W, f, fp, seed=seed, **disc_kwargs)
        lle_f = compute_lle_continuous(W, f, fp, tau=tau, seed=seed, **flow_kwargs)
        if np.isfinite(lle_m):  lles_map.append(lle_m)
        if np.isfinite(lle_f):  lles_flow.append(lle_f)
    return (
        np.mean(lles_map) if lles_map else np.nan,
        np.std(lles_map)  if lles_map else np.nan,
        np.mean(lles_flow) if lles_flow else np.nan,
        np.std(lles_flow)  if lles_flow else np.nan,
    )


# ============================================================
# Sweep A — LLE vs g
# ============================================================

def sweep_A(n=50, act_kind="tanh", gain=1.0, tau=1.0, n_seeds=5,
            disc_kwargs=None, flow_kwargs=None, verbose=True):
    """
    Primary sweep: LLE vs spectral radius g for both formulations.

    Parameters
    ----------
    n         : network size
    act_kind  : activation function name
    gain      : activation gain
    tau       : time constant for continuous flow
    n_seeds   : number of random W realisations per point
    disc_kwargs / flow_kwargs : passed through to compute_lle_*

    Returns DataFrame with columns:
        g, lle_map_mean, lle_map_std, lle_flow_mean, lle_flow_std
    """
    disc_kwargs = disc_kwargs or {}
    flow_kwargs = flow_kwargs or {}
    records = []

    for i, g in enumerate(G_VALUES):
        mm, ms, fm, fs = _lle_point(g, n, act_kind, gain, tau, n_seeds,
                                     disc_kwargs, flow_kwargs)
        records.append(dict(g=g,
                            lle_map_mean=mm,  lle_map_std=ms,
                            lle_flow_mean=fm, lle_flow_std=fs))
        if verbose:
            print(f"  Sweep A [{i+1:2d}/{len(G_VALUES)}]"
                  f"  g={g:.2f}  LLE_map={mm:+.3f}  LLE_flow={fm:+.3f}")

    return pd.DataFrame(records)


# ============================================================
# Sweep B — activation function comparison
# ============================================================

def sweep_B(n=50, tau=1.0, gain=1.0, n_seeds=5,
            disc_kwargs=None, flow_kwargs=None, verbose=True):
    """
    LLE vs g for each of the 9 activation functions.

    Returns DataFrame with columns:
        g, activation, lle_map_mean, lle_map_std, lle_flow_mean, lle_flow_std
    """
    disc_kwargs = disc_kwargs or {}
    flow_kwargs = flow_kwargs or {}
    records = []
    total = len(ALL_ACTIVATIONS) * len(G_VALUES)
    idx   = 0

    for act_kind in ALL_ACTIVATIONS:
        for g in G_VALUES:
            mm, ms, fm, fs = _lle_point(g, n, act_kind, gain, tau, n_seeds,
                                         disc_kwargs, flow_kwargs)
            records.append(dict(g=g, activation=act_kind,
                                lle_map_mean=mm,  lle_map_std=ms,
                                lle_flow_mean=fm, lle_flow_std=fs))
            idx += 1
            if verbose and idx % 20 == 0:
                print(f"  Sweep B [{idx:3d}/{total}]"
                      f"  act={act_kind:<20s}  g={g:.2f}"
                      f"  LLE_map={mm:+.3f}  LLE_flow={fm:+.3f}")

    return pd.DataFrame(records)


# ============================================================
# Sweep C — network size
# ============================================================

def sweep_C(act_kind="tanh", gain=1.0, tau=1.0, n_seeds=5,
            n_values=None, disc_kwargs=None, flow_kwargs=None, verbose=True):
    """
    LLE vs g for n ∈ {10, 50, 100, 500}.

    Returns DataFrame with columns:
        g, n, lle_map_mean, lle_map_std, lle_flow_mean, lle_flow_std
    """
    if n_values is None:
        n_values = [10, 50, 100, 200]   # 500 is feasible but slow (~8 min/seed)
    disc_kwargs = disc_kwargs or {}
    flow_kwargs = flow_kwargs or {}
    records = []
    total = len(n_values) * len(G_VALUES)
    idx   = 0

    for n in n_values:
        for g in G_VALUES:
            mm, ms, fm, fs = _lle_point(g, n, act_kind, gain, tau, n_seeds,
                                         disc_kwargs, flow_kwargs)
            records.append(dict(g=g, n=n,
                                lle_map_mean=mm,  lle_map_std=ms,
                                lle_flow_mean=fm, lle_flow_std=fs))
            idx += 1
            if verbose and idx % 10 == 0:
                print(f"  Sweep C [{idx:3d}/{total}]"
                      f"  n={n:3d}  g={g:.2f}"
                      f"  LLE_map={mm:+.3f}  LLE_flow={fm:+.3f}")

    return pd.DataFrame(records)


# ============================================================
# Sweep D — time constant τ  (continuous flow only)
# ============================================================

def sweep_D(n=50, act_kind="tanh", gain=1.0, n_seeds=5,
            tau_values=None, flow_kwargs=None, verbose=True):
    """
    LLE vs g for τ ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} (flow only).

    Returns DataFrame with columns:
        g, tau, lle_flow_mean, lle_flow_std
    """
    if tau_values is None:
        tau_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    flow_kwargs = flow_kwargs or {}
    records = []
    total = len(tau_values) * len(G_VALUES)
    idx   = 0
    f, fp = get_activation(act_kind, gain=gain)

    for tau in tau_values:
        for g in G_VALUES:
            lles_flow = []
            for seed in range(n_seeds):
                W    = init_W(n, g, seed=seed * 31 + 7)
                lle_f = compute_lle_continuous(W, f, fp, tau=tau,
                                               seed=seed, **flow_kwargs)
                if np.isfinite(lle_f):
                    lles_flow.append(lle_f)
            fm = np.mean(lles_flow) if lles_flow else np.nan
            fs = np.std(lles_flow)  if lles_flow else np.nan
            records.append(dict(g=g, tau=tau,
                                lle_flow_mean=fm, lle_flow_std=fs))
            idx += 1
            if verbose and idx % 10 == 0:
                print(f"  Sweep D [{idx:3d}/{total}]"
                      f"  τ={tau:.1f}  g={g:.2f}"
                      f"  LLE_flow={fm:+.3f}")

    return pd.DataFrame(records)
