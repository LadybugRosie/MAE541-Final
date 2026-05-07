"""
rnn_models.py
=============
Activation functions (with derivatives), weight initialisation,
discrete-map runner, and continuous-flow runner.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ============================================================
# Helper numerics
# ============================================================

def _sigmoid(x):
    x_safe = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x_safe))


def _stable_softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _gelu(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _gelu_deriv(x):
    c = np.sqrt(2.0 / np.pi)
    inner   = c * (x + 0.044715 * x ** 3)
    d_inner = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    th      = np.tanh(inner)
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th ** 2) * d_inner


# ============================================================
# Activation catalogue  {name: (f, f_prime)}
# f and f_prime operate on the *pre-scaled* input z = gain * u
# ============================================================

_ACTS = {
    "tanh": (
        lambda z: np.tanh(z),
        lambda z: 1.0 - np.tanh(z) ** 2,
    ),
    "softsign": (
        lambda z: z / (1.0 + np.abs(z)),
        lambda z: 1.0 / (1.0 + np.abs(z)) ** 2,
    ),
    "arctan": (
        lambda z: (2.0 / np.pi) * np.arctan(z),
        lambda z: (2.0 / np.pi) / (1.0 + z ** 2),
    ),
    "centered_sigmoid": (
        lambda z: 2.0 * _sigmoid(z) - 1.0,
        lambda z: 2.0 * _sigmoid(z) * (1.0 - _sigmoid(z)),
    ),
    "sigmoid_firing": (
        lambda z: _sigmoid(z),
        lambda z: _sigmoid(z) * (1.0 - _sigmoid(z)),
    ),
    "bounded_softplus": (
        lambda z: _stable_softplus(z) / (1.0 + _stable_softplus(z)),
        lambda z: _sigmoid(z) / (1.0 + _stable_softplus(z)) ** 2,
    ),
    "smooth_capped_relu": (
        lambda z: np.tanh(_stable_softplus(z)),
        lambda z: (1.0 - np.tanh(_stable_softplus(z)) ** 2) * _sigmoid(z),
    ),
    "clipped_gelu": (
        lambda z: np.clip(_gelu(z), -1.0, 1.0),
        lambda z: _gelu_deriv(z) * (np.abs(_gelu(z)) < 1.0).astype(float),
    ),
    "bounded_gelu": (
        lambda z: np.tanh(_gelu(z)),
        lambda z: _gelu_deriv(z) * (1.0 - np.tanh(_gelu(z)) ** 2),
    ),
}

ALL_ACTIVATIONS = list(_ACTS.keys())


def get_activation(kind, gain=1.0):
    """
    Return (f, f_prime) including gain scaling.

        f(u)  = f0(gain * u)
        f'(u) = gain * f0'(gain * u)
    """
    f0, fp0 = _ACTS[kind]
    f  = lambda u: f0(gain * u)
    fp = lambda u: gain * fp0(gain * u)
    return f, fp


def deriv_at_zero(kind, gain=1.0):
    """f'(0) for the named activation with given gain."""
    _, fp = get_activation(kind, gain)
    return float(fp(np.zeros(1))[0])


# ============================================================
# Weight initialisation
# ============================================================

def init_W(n, g, seed=0):
    """
    Random Gaussian weight matrix W_{ij} ~ N(0, g²/n).
    By the circular law the spectral radius ρ(W) ≈ g for large n.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, g / np.sqrt(n), (n, n))


def init_h0(n, seed=0, scale=0.01):
    """Small random initial condition."""
    return np.random.default_rng(seed).normal(0.0, scale, n)


# ============================================================
# Discrete map:  h_{t+1} = W f(h_t)
# ============================================================

def run_discrete(h0, W, f, T_steps):
    """
    Iterate the discrete map T_steps times.

    Returns H of shape (T_steps+1, n).
    Sets remaining rows to NaN on divergence.
    """
    n = len(h0)
    H = np.empty((T_steps + 1, n))
    H[0] = h0
    for t in range(T_steps):
        H[t + 1] = W @ f(H[t])
        if not np.all(np.isfinite(H[t + 1])):
            H[t + 1:] = np.nan
            break
    return H


# ============================================================
# Continuous flow:  τ ḣ = −h + W f(h)
# ============================================================

def run_continuous(h0, W, f, tau, T, dt=0.05):
    """
    Integrate the continuous-flow RNN with solve_ivp (RK45).

    Returns (t, H) where H is (T_steps, n), or (None, None) on failure.
    """
    def rhs(t, h):
        return (-h + W @ f(h)) / tau

    t_eval = np.arange(0.0, T + dt, dt)
    sol = solve_ivp(rhs, (0.0, T), h0, t_eval=t_eval,
                    method="RK45", rtol=1e-6, atol=1e-8)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None, None
    return sol.t, sol.y.T
