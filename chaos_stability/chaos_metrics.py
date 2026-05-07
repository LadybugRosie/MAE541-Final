"""
chaos_metrics.py
================
Largest Lyapunov exponent (single-vector QR / power-iteration method),
trajectory divergence, autocorrelation, and power spectral density.

LLE units
---------
  Discrete map   — nats per step
  Continuous flow — nats per time unit

Both equal zero at the edge of chaos and are positive in the chaotic regime.
"""

import numpy as np
from scipy.signal import periodogram
from rnn_models import run_discrete, run_continuous, init_h0


# ============================================================
# Fixed-step RK4 helpers  (faster than solve_ivp for many short bursts)
# ============================================================

def _rk4_h(h, W, f, tau, dt):
    """One RK4 step for τ ḣ = −h + W f(h)."""
    def rhs(h_):
        return (-h_ + W @ f(h_)) / tau
    k1 = rhs(h)
    k2 = rhs(h + 0.5 * dt * k1)
    k3 = rhs(h + 0.5 * dt * k2)
    k4 = rhs(h + dt * k3)
    return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _rk4_hv(h, v, W, f, fp, tau, dt):
    """
    One coupled RK4 step for the flow + variational equation:
        τ ḣ = −h + W f(h)
        τ v̇ = −v + W diag(f′(h)) v
    """
    def rhs_h(h_):
        return (-h_ + W @ f(h_)) / tau

    def rhs_v(h_, v_):
        return (-v_ + W @ (fp(h_) * v_)) / tau

    k1h = rhs_h(h);           k1v = rhs_v(h, v)
    h2 = h + 0.5*dt*k1h;      v2 = v + 0.5*dt*k1v
    k2h = rhs_h(h2);          k2v = rhs_v(h2, v2)
    h3 = h + 0.5*dt*k2h;      v3 = v + 0.5*dt*k2v
    k3h = rhs_h(h3);          k3v = rhs_v(h3, v3)
    h4 = h + dt*k3h;          v4 = v + dt*k3v
    k4h = rhs_h(h4);          k4v = rhs_v(h4, v4)

    h_new = h + (dt / 6.0) * (k1h + 2*k2h + 2*k3h + k4h)
    v_new = v + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return h_new, v_new


# ============================================================
# Largest Lyapunov Exponent — discrete map
# ============================================================

def compute_lle_discrete(W, f, fp,
                          n_warmup=200, n_steps=1000, seed=0):
    """
    LLE for h_{t+1} = W f(h_t) via single-vector power iteration.

    Returns float (nan on divergence, -inf if vector collapses).
    Units: nats / step.
    """
    n   = W.shape[0]
    rng = np.random.default_rng(seed)
    h   = rng.normal(0.0, 0.01, n)

    for _ in range(n_warmup):
        h = W @ f(h)
        if not np.all(np.isfinite(h)):
            return np.nan

    v = rng.normal(0.0, 1.0, n)
    v /= np.linalg.norm(v)
    log_sum = 0.0

    for _ in range(n_steps):
        fp_h  = fp(h)
        h_new = W @ f(h)
        if not np.all(np.isfinite(h_new)):
            return np.nan

        # Tangent map:  δh → W diag(f′(h_t)) δh
        Jv   = W @ (fp_h * v)
        norm = np.linalg.norm(Jv)
        if norm < 1e-300:
            return -np.inf
        log_sum += np.log(norm)
        v  = Jv / norm
        h  = h_new

    return log_sum / n_steps


# ============================================================
# Largest Lyapunov Exponent — continuous flow
# ============================================================

def compute_lle_continuous(W, f, fp, tau,
                            T_warmup=20.0, T_measure=50.0,
                            dt=0.05, n_reorth=100, seed=0):
    """
    LLE for τ ḣ = −h + W f(h) via periodic reorthogonalisation.

    Warmup and measurement use fixed-step RK4 (fast, avoids solve_ivp overhead).
    Reorthogonalises every (T_measure / n_reorth) time units.

    Returns float (nan on divergence).
    Units: nats / time unit.
    """
    n   = W.shape[0]
    rng = np.random.default_rng(seed)
    h   = rng.normal(0.0, 0.01, n)

    # ---- warmup ----
    n_warmup_steps = max(1, int(T_warmup / dt))
    for _ in range(n_warmup_steps):
        h = _rk4_h(h, W, f, tau, dt)
        if not np.all(np.isfinite(h)):
            return np.nan

    # ---- measurement ----
    v = rng.normal(0.0, 1.0, n)
    v /= np.linalg.norm(v)

    dt_reorth      = T_measure / n_reorth
    steps_per_seg  = max(1, int(dt_reorth / dt))
    # recompute actual dt_reorth to match integer steps
    dt_reorth_real = steps_per_seg * dt

    log_sum = 0.0

    for _ in range(n_reorth):
        for _ in range(steps_per_seg):
            h, v = _rk4_hv(h, v, W, f, fp, tau, dt)
            if not np.all(np.isfinite(h)) or not np.all(np.isfinite(v)):
                return np.nan

        norm = np.linalg.norm(v)
        if norm < 1e-300:
            return -np.inf
        log_sum += np.log(norm)
        v /= norm

    T_total = n_reorth * dt_reorth_real
    return log_sum / T_total


# ============================================================
# Trajectory divergence
# ============================================================

def trajectory_divergence(W, f, tau, mode,
                           T_steps=400, dt=0.05, eps=1e-6, seed=0):
    """
    Track ‖h₁(t) − h₂(t)‖ for two trajectories separated by eps at t=0.

    mode : 'discrete' or 'continuous'
    Returns (time_axis, delta) — (None, None) on failure.
    """
    n  = W.shape[0]
    h0 = init_h0(n, seed=seed)
    d0 = np.zeros(n);  d0[0] = eps

    if mode == "discrete":
        H1 = run_discrete(h0,       W, f, T_steps)
        H2 = run_discrete(h0 + d0,  W, f, T_steps)
        steps = np.arange(T_steps + 1)
        delta = np.linalg.norm(H1 - H2, axis=1)
        return steps, delta

    else:  # continuous
        T  = T_steps * dt
        t1, Y1 = run_continuous(h0,       W, f, tau, T, dt)
        t2, Y2 = run_continuous(h0 + d0,  W, f, tau, T, dt)
        if t1 is None or t2 is None:
            return None, None
        # align lengths (solve_ivp may give slightly different step counts)
        L = min(len(t1), len(t2))
        delta = np.linalg.norm(Y1[:L] - Y2[:L], axis=1)
        return t1[:L], delta


# ============================================================
# Autocorrelation
# ============================================================

def autocorrelation(x, max_lag=200):
    """
    Normalised autocorrelation  C(k) = ⟨x(t) x(t+k)⟩ / ⟨x²⟩.
    Returns array of length max_lag+1.
    """
    x   = np.asarray(x, dtype=float)
    x   = x - x.mean()
    var = np.mean(x ** 2)
    if var < 1e-14:
        return np.zeros(max_lag + 1)
    max_lag = min(max_lag, len(x) // 2)
    return np.array([
        np.mean(x[:len(x) - k] * x[k:]) / var
        for k in range(max_lag + 1)
    ])


# ============================================================
# Power spectral density
# ============================================================

def power_spectral_density(x, dt=1.0):
    """
    Welch-style PSD using scipy periodogram with Hann window.
    Returns (frequencies, Pxx).
    """
    x = np.asarray(x, dtype=float)
    f, Pxx = periodogram(x, fs=1.0 / dt, window="hann", scaling="density")
    return f, Pxx
