"""
fourier.py
==========
Oja-Hebbian network as a Fourier basis for wave approximation.

Drive an n-neuron recurrent network with a sinusoidal input at omega_drive,
fit a linear readout to reconstruct a target sine at omega_target (frequency
transformation), and sweep hyperparameter space to find configurations that
maximise R².

Sections
--------
1.  Activation functions
2.  n-neuron Oja ODE + simulator
3.  Linear readout (frequency-transformation fit)
4.  Single experiment runner
5.  Hyperparameter sweep  (random search + fine grid)
6.  Analysis plots
7.  GIF animation
8.  Main experiment pipeline  [run_fourier_experiment()]
9.  Legacy 3-neuron recurrence analysis (original code, preserved)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.patches import Patch
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ============================================================
# 1.  Activation functions
# ============================================================

def gelu_approx(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def stable_softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def activation(u, gain=2.5, kind="softsign"):
    z = gain * u
    if kind == "tanh":
        return np.tanh(z)
    elif kind == "softsign":
        return z / (1 + np.abs(z))
    elif kind == "sigmoid_firing":
        z_safe = np.clip(z, -60, 60)
        return 1.0 / (1 + np.exp(-z_safe))
    elif kind == "bounded_softplus":
        sp = stable_softplus(z)
        return sp / (1 + sp)
    elif kind == "smooth_capped_relu":
        sp = stable_softplus(z)
        return np.tanh(sp)
    elif kind == "arctan":
        return (2 / np.pi) * np.arctan(z)
    elif kind == "centered_sigmoid":
        z_safe = np.clip(z, -60, 60)
        return 2 / (1 + np.exp(-z_safe)) - 1
    elif kind == "clipped_gelu":
        return np.clip(gelu_approx(z), -1.0, 1.0)
    elif kind == "bounded_gelu":
        return np.tanh(gelu_approx(z))
    else:
        raise ValueError(f"Unknown activation: {kind}")


ALL_ACTIVATIONS = [
    "tanh",
    "softsign",
    "arctan",
    "centered_sigmoid",
    "clipped_gelu",
    "bounded_gelu",
    "sigmoid_firing",
    "bounded_softplus",
    "smooth_capped_relu",
]


# ============================================================
# 2.  n-neuron Oja ODE + simulator
# ============================================================

def oja_rhs(t, y, params):
    """
    RHS of the generalised n-neuron Oja-type recurrent system.

    State layout:
        y = [x_1 … x_n,  W_11 … W_nn]     (length  n + n²)

    Activity equation:
        τ ẋᵢ = −xᵢ + f( ∑ⱼ Wᵢⱼ xⱼ + inputᵢ(t) )
        inputᵢ(t) = A · sin(ω_drive · t + φᵢ) + bᵢ

    Oja plasticity:
        Ẇᵢⱼ = η xᵢ (xⱼ − xᵢ Wᵢⱼ) − λ Wᵢⱼ
        (zeroed when oja_frozen=True)
    """
    n           = params["n"]
    tau         = params["tau"]
    eta         = params["eta"]
    lam         = params["lambda"]
    gain        = params["gain"]
    act_kind    = params.get("activation_kind", "tanh")
    frozen      = params.get("oja_frozen", False)
    b           = np.asarray(params.get("b", np.zeros(n)))
    A           = float(params.get("A", 1.0))
    omega_drive = float(params.get("omega_drive", 1.0))
    phi         = np.asarray(params.get("phi", np.zeros(n)))

    x = y[:n]
    W = y[n:].reshape(n, n)

    drive = A * np.sin(omega_drive * t + phi) + b
    dxdt  = (-x + activation(W @ x + drive, gain=gain, kind=act_kind)) / tau

    if frozen:
        dWdt = np.zeros_like(W)
    else:
        # vectorised Oja rule: η (outer(x,x) − diag(x²) W) − λ W
        dWdt = eta * (np.outer(x, x) - (x ** 2)[:, None] * W) - lam * W

    return np.concatenate([dxdt, dWdt.ravel()])


def make_y0(n, seed=42):
    """Random initial state vector for an n-neuron Oja system."""
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.uniform(-0.3, 0.3, n),
        rng.uniform(-1.0, 1.0, (n, n)).ravel(),
    ])


def default_params(n=3):
    return {
        "n":               n,
        "tau":             1.0,
        "eta":             0.04,
        "lambda":          0.004,
        "b":               np.zeros(n),
        "gain":            3.0,
        "activation_kind": "softsign",
        "A":               1.0,
        "omega_drive":     1.0,
        "phi":             np.zeros(n),
        "oja_frozen":      False,
    }


def simulate_oja_driven(T=200, dt=0.05, params=None, y0=None, seed=42):
    """
    Integrate the n-neuron Oja system with sinusoidal driving input.

    Returns
    -------
    t      : (T_steps,)           time vector
    Y      : (T_steps, n + n²)    full state history (activities first, then weights)
    params : dict                 (passed through unchanged)
    """
    if params is None:
        params = default_params()
    n = params["n"]
    if y0 is None:
        y0 = make_y0(n, seed=seed)

    t_eval = np.arange(0, T + dt, dt)
    sol = solve_ivp(
        fun=lambda t, y: oja_rhs(t, y, params),
        t_span=(0, T),
        y0=y0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T, params


# ============================================================
# 3.  Linear readout  (frequency-transformation fit)
# ============================================================

def fit_linear_readout(t, x_activities, omega_target, transient_frac=0.3):
    """
    Solve  c* = argmin ‖X c − sin(ω_target t)‖²  (least squares).

    Parameters
    ----------
    t             : (T_steps,)    full time vector
    x_activities  : (T_steps, n)  neuron activities only
    omega_target  : float         target reconstruction frequency
    transient_frac: float         fraction of trajectory to discard as transient

    Returns
    -------
    c      : (n,)    readout weights
    r2     : float   coefficient of determination
    nrmse  : float   normalised RMSE  (RMSE / signal range)
    y_pred : (T_ss,) reconstructed signal (steady-state window)
    t_ss   : (T_ss,) steady-state time vector
    target : (T_ss,) target signal
    """
    n_trans = int(len(t) * transient_frac)
    t_ss    = t[n_trans:]
    X_ss    = x_activities[n_trans:]
    target  = np.sin(omega_target * t_ss)

    c, *_  = np.linalg.lstsq(X_ss, target, rcond=None)
    y_pred = X_ss @ c

    ss_tot = np.sum((target - target.mean()) ** 2)
    ss_res = np.sum((target - y_pred) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    nrmse  = float(
        np.sqrt(np.mean((target - y_pred) ** 2))
        / max(target.max() - target.min(), 1e-10)
    )
    return c, r2, nrmse, y_pred, t_ss, target


# ============================================================
# 4.  Single experiment runner
# ============================================================

def run_experiment(params, omega_target, T=150, dt=0.1,
                   transient_frac=0.3, seed=42):
    """
    Simulate + fit readout for one parameter configuration.

    Returns a dict with keys {r2, nrmse, c, t, Y, t_ss, y_pred, target},
    or None if the simulation diverges or raises.
    """
    try:
        t, Y, _ = simulate_oja_driven(T=T, dt=dt, params=params, seed=seed)
        if not np.all(np.isfinite(Y)):
            return None
        n = params["n"]
        c, r2, nrmse, y_pred, t_ss, target = fit_linear_readout(
            t, Y[:, :n], omega_target, transient_frac=transient_frac
        )
        return dict(r2=r2, nrmse=nrmse, c=c,
                    t=t, Y=Y, t_ss=t_ss, y_pred=y_pred, target=target)
    except Exception:
        return None


# ============================================================
# 5.  Hyperparameter sweep  (random search + fine grid)
# ============================================================

PARAM_RANGES = {
    "n":               [3, 5, 8, 12],
    "omega_drive":     [0.1, 0.5, 1.0, 2.0, 5.0],
    "omega_target":    [0.1, 0.5, 1.0, 2.0, 5.0],
    "oja_frozen":      [True, False],
    "tau":             [0.2, 0.5, 1.0, 2.0, 5.0],
    "eta":             [0.001, 0.01, 0.05, 0.1],
    "lambda":          [0.0001, 0.001, 0.01],
    "gain":            [1.0, 2.0, 3.0, 5.0],
    "activation_kind": ALL_ACTIVATIONS,
}


def random_sweep(n_samples=500, param_ranges=None, T=150, dt=0.1,
                 transient_frac=0.3, seed=0, verbose=True):
    """
    Uniform random search over PARAM_RANGES.

    Returns a DataFrame — one row per successful run — with columns:
    n, tau, eta, lambda, gain, activation_kind, oja_frozen,
    omega_drive, omega_target, r2, nrmse, phi_seed.
    """
    if param_ranges is None:
        param_ranges = PARAM_RANGES
    rng     = np.random.default_rng(seed)
    records = []

    for i in range(n_samples):
        sampled      = {k: rng.choice(v) for k, v in param_ranges.items()}
        n            = int(sampled["n"])
        omega_target = float(sampled.pop("omega_target"))

        phi_seed = int(rng.integers(100_000))
        phi      = np.random.default_rng(phi_seed).uniform(0, 2 * np.pi, n)

        params = {
            "n":               n,
            "tau":             float(sampled["tau"]),
            "eta":             float(sampled["eta"]),
            "lambda":          float(sampled["lambda"]),
            "gain":            float(sampled["gain"]),
            "activation_kind": str(sampled["activation_kind"]),
            "oja_frozen":      bool(sampled["oja_frozen"]),
            "A":               1.0,
            "omega_drive":     float(sampled["omega_drive"]),
            "b":               np.zeros(n),
            "phi":             phi,
        }

        sim_seed = int(rng.integers(100_000))
        result   = run_experiment(params, omega_target, T=T, dt=dt,
                                  transient_frac=transient_frac, seed=sim_seed)
        if result is not None:
            records.append({
                "n":               n,
                "tau":             params["tau"],
                "eta":             params["eta"],
                "lambda":          params["lambda"],
                "gain":            params["gain"],
                "activation_kind": params["activation_kind"],
                "oja_frozen":      params["oja_frozen"],
                "omega_drive":     params["omega_drive"],
                "omega_target":    omega_target,
                "r2":              result["r2"],
                "nrmse":           result["nrmse"],
                "phi_seed":        phi_seed,
                "sim_seed":        sim_seed,
            })

        if verbose and (i + 1) % 50 == 0:
            best = max(r["r2"] for r in records) if records else float("nan")
            print(f"  [{i+1:4d}/{n_samples}]  ok={len(records):4d}  best_r2={best:.4f}")

    return pd.DataFrame(records)


def fine_grid_sweep(best_row, T=200, dt=0.05, transient_frac=0.3,
                    n_phi_seeds=5, n_grid=4, verbose=True):
    """
    Dense grid in (τ, η, λ, gain) centred on the best random-search row,
    averaged over n_phi_seeds random phase realisations.

    Returns a DataFrame with r2_mean, r2_std per grid point.
    """
    def neighbourhood(v, factor=2.0):
        return np.linspace(v / factor, v * factor, n_grid)

    tau_vals  = neighbourhood(float(best_row["tau"]))
    eta_vals  = neighbourhood(float(best_row["eta"]))
    lam_vals  = neighbourhood(float(best_row["lambda"]))
    gain_vals = neighbourhood(float(best_row["gain"]))

    n            = int(best_row["n"])
    omega_drive  = float(best_row["omega_drive"])
    omega_target = float(best_row["omega_target"])
    act_kind     = str(best_row["activation_kind"])
    frozen       = bool(best_row["oja_frozen"])

    combos  = [(tau, eta, lam, gain)
               for tau  in tau_vals
               for eta  in eta_vals
               for lam  in lam_vals
               for gain in gain_vals]
    records = []

    for idx, (tau, eta, lam, gain) in enumerate(combos):
        r2s = []
        for phi_seed in range(n_phi_seeds):
            phi    = np.random.default_rng(phi_seed * 137).uniform(0, 2 * np.pi, n)
            params = {
                "n": n, "tau": tau, "eta": eta, "lambda": lam, "gain": gain,
                "activation_kind": act_kind, "oja_frozen": frozen,
                "A": 1.0, "omega_drive": omega_drive,
                "b": np.zeros(n), "phi": phi,
            }
            res = run_experiment(params, omega_target, T=T, dt=dt,
                                 transient_frac=transient_frac, seed=phi_seed)
            if res is not None:
                r2s.append(res["r2"])

        if r2s:
            records.append(dict(tau=tau, eta=eta, lam=lam, gain=gain,
                                r2_mean=np.mean(r2s), r2_std=np.std(r2s),
                                n_ok=len(r2s)))

        if verbose and (idx + 1) % 20 == 0:
            print(f"  fine grid [{idx+1}/{len(combos)}]")

    return pd.DataFrame(records)


# ============================================================
# 6.  Analysis plots
# ============================================================

def _draw_heatmap(pivot, ax, label="R²"):
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.3g}" for v in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.3g}" for v in pivot.index])
    ax.get_figure().colorbar(im, ax=ax, label=label)
    return im


def plot_r2_heatmap(df, x_col, y_col, agg="median", ax=None):
    """Heatmap of aggregated R² over two hyperparameter axes."""
    pivot = df.pivot_table(values="r2", index=y_col, columns=x_col, aggfunc=agg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{agg} R²:  {x_col}  vs  {y_col}")
    _draw_heatmap(pivot, ax)
    return ax.get_figure(), ax


def plot_frequency_matrix(df):
    """5×5 heatmap of median R² over (omega_drive, omega_target) pairs,
    split into Oja-learning and Oja-frozen panels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, frozen_val, lbl in zip(axes, [False, True], ["Oja learning", "Oja frozen"]):
        sub = df[df["oja_frozen"] == frozen_val]
        if sub.empty:
            ax.set_title(f"{lbl}: no data"); continue
        pivot = sub.pivot_table(values="r2", index="omega_target",
                                columns="omega_drive", aggfunc="median")
        ax.set_xlabel("ω_drive"); ax.set_ylabel("ω_target"); ax.set_title(lbl)
        _draw_heatmap(pivot, ax)
    fig.suptitle("Frequency-transformation R² matrix")
    fig.tight_layout()
    return fig


def plot_activation_boxplot(df):
    """Box plot of R² per activation function, split by plasticity mode."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, frozen_val, lbl in zip(axes, [False, True], ["Oja learning", "Oja frozen"]):
        sub  = df[df["oja_frozen"] == frozen_val]
        data = [sub[sub["activation_kind"] == a]["r2"].dropna().values
                for a in ALL_ACTIVATIONS]
        ax.boxplot(data, labels=ALL_ACTIVATIONS)
        ax.set_xticklabels(ALL_ACTIVATIONS, rotation=45, ha="right")
        ax.set_ylabel("R²"); ax.set_title(lbl)
        ax.axhline(0.9, linestyle="--", color="red", alpha=0.5, label="R²=0.9")
        ax.legend()
    fig.suptitle("R² distribution by activation function")
    fig.tight_layout()
    return fig


def plot_neuron_count_effect(df):
    """Side-by-side box plots of R² vs n for learning and frozen modes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ns      = sorted(df["n"].unique())
    handles = []
    for frozen_val, lbl, color, offset in [
        (False, "Oja learning", "steelblue", -0.15),
        (True,  "Oja frozen",   "tomato",     0.15),
    ]:
        sub  = df[df["oja_frozen"] == frozen_val]
        data = [sub[sub["n"] == nv]["r2"].dropna().values for nv in ns]
        ax.boxplot(data,
                   positions=[i + offset for i in range(len(ns))],
                   widths=0.25, patch_artist=True,
                   boxprops=dict(facecolor=color),
                   medianprops=dict(color="black"))
        handles.append(Patch(facecolor=color, label=lbl))
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(ns)
    ax.set_xlabel("number of neurons (n)")
    ax.set_ylabel("R²")
    ax.set_title("Effect of neuron count on reconstruction quality")
    ax.legend(handles=handles)
    fig.tight_layout()
    return fig


def plot_best_reconstruction(result, params, omega_target,
                             title="Best reconstruction"):
    """Static overlay of target and reconstructed signal with residual."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    t_ss, target, y_pred = result["t_ss"], result["target"], result["y_pred"]
    ax1.plot(t_ss, target, "b-",  lw=1.5, label="target  sin(ω_target · t)")
    ax1.plot(t_ss, y_pred, "r--", lw=1.5,
             label=f"reconstruction   R²={result['r2']:.4f}")
    ax1.set_xlabel("time"); ax1.set_ylabel("amplitude")
    ax1.legend(); ax1.set_title(title)
    ax2.plot(t_ss, target - y_pred, "k-", lw=1, alpha=0.7)
    ax2.axhline(0, linestyle="--", color="gray")
    ax2.set_xlabel("time"); ax2.set_ylabel("residual")
    ax2.set_title("Reconstruction residual")
    fig.tight_layout()
    return fig


# ============================================================
# 7.  GIF animation
# ============================================================

def animate_reconstruction(
    result,
    params,
    omega_target,
    filename="best_reconstruction.gif",
    fps=15,
    window_periods=3,
    compare_result=None,
    compare_label="Oja frozen",
):
    """
    Sliding-window GIF of reconstruction vs target.

    Layout (per row):
      Left  — full signal overview with orange vertical markers for the window
      Right — zoomed window with local R² in the title

    When compare_result is provided a second row is added (learning vs frozen).
    """
    T_target    = 2 * np.pi / max(omega_target, 1e-6)
    t_ss_ref    = result["t_ss"]
    dt_val      = float(t_ss_ref[1] - t_ss_ref[0]) if len(t_ss_ref) > 1 else 0.1
    window_size = max(10, int(window_periods * T_target / dt_val))
    step        = max(1,  int(0.25 * T_target / dt_val))
    starts      = list(range(0, len(t_ss_ref) - window_size, step))

    if not starts:
        print("Time series too short for GIF — skipping.")
        return

    results_list = [result]
    labels_list  = [f"Oja learning  (R²={result['r2']:.3f})"]
    if compare_result is not None:
        results_list.append(compare_result)
        labels_list.append(f"{compare_label}  (R²={compare_result['r2']:.3f})")

    n_rows = len(results_list)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), squeeze=False)

    vline_los, vline_his, zoom_lt, zoom_lp = [], [], [], []

    for row, (res, lbl) in enumerate(zip(results_list, labels_list)):
        ax_ov   = axes[row, 0]
        ax_zoom = axes[row, 1]

        ax_ov.plot(res["t_ss"], res["target"], "b-",  alpha=0.2, lw=0.8)
        ax_ov.plot(res["t_ss"], res["y_pred"], "r-",  alpha=0.2, lw=0.8)
        x0 = res["t_ss"][0]
        x1 = res["t_ss"][min(window_size - 1, len(res["t_ss"]) - 1)]
        vlo = ax_ov.axvline(x0, color="orange", lw=1.5)
        vhi = ax_ov.axvline(x1, color="orange", lw=1.5, linestyle="--")
        ax_ov.set_title(lbl); ax_ov.set_xlabel("time")

        lt, = ax_zoom.plot([], [], "b-",  lw=2, label="target")
        lp, = ax_zoom.plot([], [], "r--", lw=2, label="reconstruction")
        ax_zoom.set_ylim(-1.6, 1.6)
        ax_zoom.legend(fontsize=8); ax_zoom.set_xlabel("time")

        vline_los.append(vlo); vline_his.append(vhi)
        zoom_lt.append(lt);    zoom_lp.append(lp)

    sup_title = (f"ω_drive={params['omega_drive']:.2f}  "
                 f"ω_target={omega_target:.2f}  "
                 f"n={params['n']}  "
                 f"act={params['activation_kind']}")
    fig.suptitle(sup_title)
    fig.tight_layout()

    def update(frame_idx):
        start = starts[frame_idx]
        end   = start + window_size
        for row, res in enumerate(results_list):
            N    = len(res["t_ss"])
            t_w  = res["t_ss"]   [start:min(end, N)]
            tg_w = res["target"] [start:min(end, N)]
            pr_w = res["y_pred"] [start:min(end, N)]
            if len(t_w) == 0:
                continue
            zoom_lt[row].set_data(t_w, tg_w)
            zoom_lp[row].set_data(t_w, pr_w)
            axes[row, 1].set_xlim(t_w[0], t_w[-1])
            vline_los[row].set_xdata([t_w[0],  t_w[0]])
            vline_his[row].set_xdata([t_w[-1], t_w[-1]])
            ss_tot   = np.sum((tg_w - tg_w.mean()) ** 2)
            ss_res   = np.sum((tg_w - pr_w) ** 2)
            local_r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
            axes[row, 1].set_title(f"t = {t_w[0]:.1f}   local R² = {local_r2:.3f}")

    ani = animation.FuncAnimation(
        fig, update, frames=len(starts), blit=False, interval=int(1000 / fps)
    )
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF: {filename}")


# ============================================================
# 8.  Main experiment pipeline
# ============================================================

def run_fourier_experiment(
    n_random=500,
    T_sweep=150,
    dt_sweep=0.1,
    T_final=300,
    dt_final=0.05,
    gif_filename="best_reconstruction.gif",
    param_ranges=None,
    seed=0,
    verbose=True,
):
    """
    Full pipeline:
      1. Random hyperparameter search         →  random_df
      2. Fine grid around best configuration  →  fine_df
      3. Save analysis PNG plots
      4. Save GIF (Oja learning vs Oja frozen, side-by-side)

    Parameters
    ----------
    n_random      : int    number of random samples in Phase 1
    T_sweep       : float  simulation length used during the sweep
    dt_sweep      : float  timestep used during the sweep
    T_final       : float  simulation length for the high-res best-config run
    dt_final      : float  timestep for the high-res run
    gif_filename  : str    output path for the animation
    param_ranges  : dict   override PARAM_RANGES (None = use defaults)
    seed          : int    master RNG seed
    verbose       : bool   print progress

    Returns
    -------
    (random_df, fine_df, best_params, best_result_learning)
    """

    # ------------------------------------------------------------------
    # Phase 1 — Random search
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Phase 1 — Random hyperparameter search")
    print("=" * 70)

    random_df = random_sweep(
        n_samples=n_random, param_ranges=param_ranges,
        T=T_sweep, dt=dt_sweep, seed=seed, verbose=verbose,
    )

    if random_df.empty:
        print("No successful runs — check parameter ranges.")
        return None, None, None, None

    print(f"\nRandom search complete.  {len(random_df)} successful runs.")
    print(f"  best R²   = {random_df['r2'].max():.4f}")
    print(f"  median R² = {random_df['r2'].median():.4f}")

    # ------------------------------------------------------------------
    # Phase 2 — Fine grid
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 2 — Fine grid around best configuration")
    print("=" * 70)

    best_row = random_df.loc[random_df["r2"].idxmax()]
    print("Best random config:")
    print(best_row.to_string())

    fine_df = fine_grid_sweep(best_row, T=T_final, dt=dt_final, verbose=verbose)
    if not fine_df.empty:
        best_fine = fine_df.loc[fine_df["r2_mean"].idxmax()]
        print(f"\nBest fine-grid R² (mean over seeds) = {best_fine['r2_mean']:.4f}")

    # ------------------------------------------------------------------
    # Phase 3 — Analysis plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 3 — Saving analysis plots")
    print("=" * 70)

    named_figs = [
        ("freq_matrix.png",        plot_frequency_matrix(random_df)),
        ("activation_boxplot.png", plot_activation_boxplot(random_df)),
        ("neuron_count.png",       plot_neuron_count_effect(random_df)),
    ]
    for fname, fig in named_figs:
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")

    for x_col, y_col in [("tau", "omega_drive"), ("eta", "lambda"), ("gain", "n")]:
        if x_col in random_df.columns and y_col in random_df.columns:
            fig, _ = plot_r2_heatmap(random_df, x_col, y_col)
            fname  = f"heatmap_{x_col}_vs_{y_col}.png"
            fig.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fname}")

    # ------------------------------------------------------------------
    # Phase 4 — High-res simulation + GIF
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 4 — GIF animation")
    print("=" * 70)

    n            = int(best_row["n"])
    omega_target = float(best_row["omega_target"])
    phi          = np.random.default_rng(int(best_row["phi_seed"])).uniform(
                       0, 2 * np.pi, n)

    best_params = {
        "n":               n,
        "tau":             float(best_row["tau"]),
        "eta":             float(best_row["eta"]),
        "lambda":          float(best_row["lambda"]),
        "gain":            float(best_row["gain"]),
        "activation_kind": str(best_row["activation_kind"]),
        "A":               1.0,
        "omega_drive":     float(best_row["omega_drive"]),
        "b":               np.zeros(n),
        "phi":             phi,
    }

    print("Running high-res simulation (Oja learning)…")
    result_learning = run_experiment(
        {**best_params, "oja_frozen": False},
        omega_target, T=T_final, dt=dt_final,
        seed=int(best_row["sim_seed"]),
    )

    print("Running high-res simulation (Oja frozen)…")
    result_frozen = run_experiment(
        {**best_params, "oja_frozen": True},
        omega_target, T=T_final, dt=dt_final,
        seed=int(best_row["sim_seed"]),
    )

    if result_learning is None:
        print("High-res learning simulation failed.")
        return random_df, fine_df, best_params, None

    fig = plot_best_reconstruction(
        result_learning, best_params, omega_target,
        title=(f"Best reconstruction (Oja learning)  "
               f"R²={result_learning['r2']:.4f}")
    )
    fig.savefig("best_reconstruction_static.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: best_reconstruction_static.png")

    animate_reconstruction(
        result=result_learning,
        params={**best_params, "oja_frozen": False},
        omega_target=omega_target,
        filename=gif_filename,
        compare_result=result_frozen,
        compare_label="Oja frozen",
    )

    print("\nAll done.")
    return random_df, fine_df, best_params, result_learning


# ============================================================
# 9.  Legacy 3-neuron recurrence-domain analysis  (original code)
# ============================================================

def oja_three_neuron_rhs(t, y, params):
    """Original 3-neuron Oja RHS — kept for backward compatibility."""
    tau  = params["tau"];  eta = params["eta"];  lam = params["lambda"]
    b    = np.array(params["b"]);  gain = params["gain"];  n = 3
    x    = y[:n];  W = y[n:].reshape(n, n)
    dxdt = (-x + activation(W @ x + b, gain=gain,
                            kind=params.get("activation_kind", "tanh"))) / tau
    dWdt = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dWdt[i, j] = eta * x[i] * (x[j] - x[i] * W[i, j]) - lam * W[i, j]
    return np.concatenate([dxdt, dWdt.ravel()])


def simulate_oja_network(T=300, dt=0.05, params=None, y0=None):
    n = 3
    if params is None:
        params = {"tau": 1.0, "eta": 0.04, "lambda": 0.004,
                  "b": [0.1, -0.1, 0.05], "gain": 3.0,
                  "activation_kind": "softsign"}
    if y0 is None:
        y0 = np.concatenate([
            np.array([0.2, -0.15, 0.1]),
            np.array([[1.2, -2.0, 0.7], [2.0, 1.2, -1.1],
                      [-0.8, 1.4, 0.9]]).ravel(),
        ])
    t_eval = np.arange(0, T + dt, dt)
    sol = solve_ivp(
        fun=lambda t, y: oja_three_neuron_rhs(t, y, params),
        t_span=(0, T), y0=y0, t_eval=t_eval,
        method="DOP853", rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T, params


def zscore(X):
    X = np.asarray(X)
    mu = X.mean(axis=0); sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma


def recurrence_symbols(X, epsilon):
    D = squareform(pdist(X, metric="euclidean"))
    R = D <= epsilon
    _, labels = connected_components(csr_matrix(R), directed=False,
                                     return_labels=True)
    mapping = {}; symbols = np.zeros_like(labels); nxt = 0
    for k, lab in enumerate(labels):
        if lab not in mapping:
            mapping[lab] = nxt; nxt += 1
        symbols[k] = mapping[lab]
    return symbols, R.astype(int), D


def entropy_ratio(symbols):
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p + 1e-12)) / max(len(counts), 1)


def choose_epsilon(X, n_eps=80, q_min=0.01, q_max=0.35):
    D  = squareform(pdist(X, metric="euclidean"))
    ev = np.quantile(D[D > 0], np.linspace(q_min, q_max, n_eps))
    scores, all_sym, all_R = [], [], []
    for eps in ev:
        sym, R, _ = recurrence_symbols(X, eps)
        scores.append(entropy_ratio(sym)); all_sym.append(sym); all_R.append(R)
    scores = np.array(scores); best = np.argmax(scores)
    return dict(best_epsilon=ev[best], best_symbols=all_sym[best],
                best_R=all_R[best], eps_values=ev, scores=scores)


def symbolic_recurrence_plot(symbols):
    symbols = np.asarray(symbols)
    return (symbols[:, None] == symbols[None, :]).astype(int)


def pca_2d(X):
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def run_analysis_for_activation(activation_kind, T=300, dt=0.05,
                                 stride=5, n_eps=50, show_recurrence=True):
    """Legacy single-activation recurrence analysis with plots."""
    print("\n" + "=" * 90)
    print(f"ACTIVATION FUNCTION: {activation_kind}")
    print("=" * 90)
    params = {"tau": 1.0, "eta": 0.04, "lambda": 0.004,
              "b": [0.1, -0.1, 0.05], "gain": 3.0,
              "activation_kind": activation_kind}
    y0 = np.concatenate([np.array([0.2, -0.15, 0.1]),
                         np.array([[1.2, -2.0, 0.7],
                                   [2.0, 1.2, -1.1],
                                   [-0.8, 1.4, 0.9]]).ravel()])
    try:
        t, Y, params = simulate_oja_network(T=T, dt=dt, params=params, y0=y0)
    except RuntimeError as e:
        print(f"Simulation failed: {e}"); return
    if not np.all(np.isfinite(Y)):
        print("Non-finite values."); return

    t_rec = t[::stride]; Y_rec = Y[::stride]
    X_norm = zscore(Y_rec)
    res    = choose_epsilon(X_norm, n_eps=n_eps)
    symbols = res["best_symbols"]; R = res["best_R"]
    R_sym   = symbolic_recurrence_plot(symbols)

    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.plot(t, Y[:, i], label=rf"$x_{i+1}$")
    plt.xlabel("time"); plt.ylabel("neural activity")
    plt.title(f"Activities | {activation_kind}"); plt.legend()
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(11, 5))
    wlbls = [rf"$W_{{{i+1}{j+1}}}$" for i in range(3) for j in range(3)]
    for k in range(9):
        plt.plot(t, Y[:, 3 + k], label=wlbls[k])
    plt.xlabel("time"); plt.ylabel("weight")
    plt.title(f"Weights | {activation_kind}")
    plt.legend(ncol=3, fontsize=9); plt.tight_layout(); plt.show()

    if show_recurrence:
        for mat, title in [(R, "ε-recurrence"), (R_sym, "symbolic recurrence")]:
            plt.figure(figsize=(6, 6))
            plt.imshow(mat, origin="lower", cmap="gray_r", interpolation="nearest")
            plt.title(f"{title} | {activation_kind}")
            plt.tight_layout(); plt.show()

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(Y_rec[:, 0], Y_rec[:, 1], Y_rec[:, 2],
               c=symbols, s=12, cmap="tab20")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_zlabel(r"$x_3$")
    ax.set_title(f"Recurrence domains | {activation_kind}")
    plt.tight_layout(); plt.show()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # --- Fourier approximation experiment (main new experiment) ---
    random_df, fine_df, best_params, best_result = run_fourier_experiment(
        n_random=500,
        T_sweep=150,
        dt_sweep=0.1,
        T_final=300,
        dt_final=0.05,
        gif_filename="best_reconstruction.gif",
        seed=0,
        verbose=True,
    )

    # --- Optional: legacy recurrence analysis (original behaviour) ---
    # for act in ALL_ACTIVATIONS:
    #     run_analysis_for_activation(act, T=300, dt=0.05)
