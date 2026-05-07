"""
oja_core.py
===========
Shared simulation utilities for the Oja-chaos experiments.

simulate_oja : integrate the Oja-driven recurrent network with fixed-step numerics.
compute_r2   : fit a linear readout and return R².
"""

import numpy as np
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'chaos_stability'))

from rnn_models import get_activation


def simulate_oja(
    W0, params, T=300.0, dt=0.05,
    rho_sample_every=20, seed=0
):
    """
    Integrate the Oja-driven recurrent network.

    Activity (RK4):   τẋ = -x + f(Wx + A·sin(ω·t + φ))
    Weights (Euler):  dW/dt = η(outer(x,x) − diag(x²)W) − λW

    The Oja rule is the same vectorised form used in fourier/fourier.py:
        dW_ij/dt = η · x_i · (x_j − x_i · W_ij) − λ · W_ij

    Parameters
    ----------
    W0               : (n, n)  initial weight matrix (copied, not modified)
    params           : dict    tau, eta, lambda, gain, act_kind,
                               A, omega_drive, oja_frozen
    T, dt            : float   total simulation time and step size
    rho_sample_every : int     sample ρ(W) every this many steps
    seed             : int     RNG seed for x₀ and per-neuron phase offsets

    Returns
    -------
    t_act    : (n_steps,)    time array
    x_hist   : (n_steps, n) activity history  (NaN rows after divergence)
    rho_hist : (K,)         spectral radius ρ(W) = max|eigenvalue(W)| at sample times
    t_rho    : (K,)         time stamps for each spectral radius sample
    W_final  : (n, n)       weight matrix at end of simulation
    """
    n           = W0.shape[0]
    tau         = params.get('tau', 1.0)
    eta         = params.get('eta', 0.01)
    lam         = params.get('lambda', 0.001)
    gain        = params.get('gain', 1.0)
    act_kind    = params.get('act_kind', 'tanh')
    A           = params.get('A', 1.0)
    omega_drive = params.get('omega_drive', 1.0)
    frozen      = params.get('oja_frozen', False)

    f, _ = get_activation(act_kind, gain=gain)

    rng = np.random.default_rng(seed)
    x   = rng.normal(0.0, 0.01, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)   # per-neuron phase offset
    W   = W0.copy().astype(float)

    n_steps = int(round(T / dt))
    t_act   = np.arange(n_steps) * dt
    x_hist  = np.empty((n_steps, n))

    rho_list   = []
    t_rho_list = []

    def _rhs(x_, t_):
        drive = A * np.sin(omega_drive * t_ + phi)
        return (-x_ + f(W @ x_ + drive)) / tau

    for i in range(n_steps):
        t = i * dt

        if i % rho_sample_every == 0:
            rho_list.append(float(np.abs(np.linalg.eigvals(W)).max()))
            t_rho_list.append(t)

        x_hist[i] = x

        k1 = _rhs(x, t)
        k2 = _rhs(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = _rhs(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = _rhs(x + dt * k3, t + dt)
        x_new = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.all(np.isfinite(x_new)):
            x_hist[i + 1:] = np.nan
            break

        if not frozen:
            dW = eta * (np.outer(x, x) - (x ** 2)[:, None] * W) - lam * W
            W  = W + dt * dW

        x = x_new

    return t_act, x_hist, np.array(rho_list), np.array(t_rho_list), W


def compute_r2(t, x_hist, omega_target, transient_frac=0.3):
    """
    Fit c* = argmin ||X c − sin(ω_target · t)||² and return R².

    Returns NaN if any activity values are non-finite.
    R² < 0 is possible (fit worse than predicting the mean).
    """
    n_trans = int(len(t) * transient_frac)
    t_ss    = t[n_trans:]
    X_ss    = x_hist[n_trans:]

    if not np.all(np.isfinite(X_ss)):
        return np.nan

    target = np.sin(omega_target * t_ss)
    c, *_  = np.linalg.lstsq(X_ss, target, rcond=None)
    y_pred = X_ss @ c

    ss_tot = float(np.sum((target - target.mean()) ** 2))
    ss_res = float(np.sum((target - y_pred) ** 2))
    return (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
