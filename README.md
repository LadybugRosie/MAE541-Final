# MAE541 Final — Oja-Hebbian Network as a Fourier Basis

## Overview

This project tests whether a recurrent network of neurons governed by **Oja-type Hebbian plasticity** can approximate a target sine wave via a weighted linear combination of its outputs — analogous to Fourier decomposition.

The network is driven with a sinusoidal input at frequency `omega_drive`. The resulting neuron activities form a basis. A least-squares linear readout is fitted to reconstruct `sin(omega_target * t)`, where `omega_target` may **differ** from `omega_drive` (frequency transformation). A hyperparameter sweep maps where in parameter space this reconstruction is accurate, and a GIF animation shows the best result varying in time.

---

## Theory

### Network dynamics

Each neuron's activity evolves as:

```
τ ẋᵢ = −xᵢ + f( ∑ⱼ Wᵢⱼ xⱼ + inputᵢ(t) )
inputᵢ(t) = A · sin(ω_drive · t + φᵢ) + bᵢ
```

The weight matrix evolves via the **Oja plasticity rule**:

```
Ẇᵢⱼ = η xᵢ (xⱼ − xᵢ Wᵢⱼ) − λ Wᵢⱼ
```

The Oja rule is a Hebbian rule with a decay term that prevents weight blow-up and encourages the principal-component structure of the input.

### Linear readout

After discarding the transient (first 30% of simulation time), the best linear combination is found by solving:

```
c* = argmin ‖ X c − sin(ω_target · t) ‖²
```

where `X` is the matrix of steady-state neuron activities. Quality is reported as R² and normalised RMSE.

### Frequency transformation

When `omega_drive ≠ omega_target` the network must internally generate a frequency component that was not directly in the input. The `(omega_drive, omega_target)` heatmap in the output plots shows where this works.

---

## File structure

```
MAE541_Final/
├── fourier.py              # all code
├── README.md               # this file
├── pyproject.toml          # uv project + dependency spec
├── uv.lock                 # locked dependency versions
├── .venv/                  # uv-managed virtual environment (do not commit)
│
│   (generated on run)
├── freq_matrix.png         # R² heatmap: omega_drive × omega_target
├── activation_boxplot.png  # R² per activation function
├── neuron_count.png        # R² vs number of neurons
├── heatmap_tau_vs_omega_drive.png
├── heatmap_eta_vs_lambda.png
├── heatmap_gain_vs_n.png
├── best_reconstruction_static.png   # static overlay of best fit
└── best_reconstruction.gif          # sliding-window animation
```

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for environment management.  A `pyproject.toml` and `.venv` are already initialised in the project directory.

```bash
# Install uv if not already present
brew install uv        # macOS
# or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (first time, or after adding packages)
uv sync
```

To add a new dependency:

```bash
uv add <package>
```

---

## Running

```bash
uv run python fourier.py
```

The script runs the full four-phase pipeline with default settings (500 random samples). To run a quick test with fewer samples, edit the entry point at the bottom of `fourier.py`:

```python
run_fourier_experiment(n_random=50, T_sweep=80, dt_sweep=0.2)
```

---

## Key functions

| Function | Purpose |
|---|---|
| `run_fourier_experiment(...)` | Full pipeline — call this to run everything |
| `random_sweep(n_samples, ...)` | Phase 1: random hyperparameter search |
| `fine_grid_sweep(best_row, ...)` | Phase 2: dense grid around best config |
| `run_experiment(params, omega_target, ...)` | Single simulate + fit run |
| `fit_linear_readout(t, x, omega_target)` | Least-squares readout, returns R² |
| `animate_reconstruction(result, ...)` | Save sliding-window GIF |
| `plot_frequency_matrix(df)` | omega_drive × omega_target R² heatmap |
| `run_analysis_for_activation(act)` | Legacy recurrence-domain analysis |

---

## Parameters reference

### Network parameters (passed as `params` dict)

| Key | Type | Default | Description |
|---|---|---|---|
| `n` | int | 3 | Number of neurons |
| `tau` | float | 1.0 | Neural time constant |
| `eta` | float | 0.04 | Oja plasticity learning rate |
| `lambda` | float | 0.004 | Weight decay in Oja rule |
| `gain` | float | 3.0 | Activation function gain |
| `activation_kind` | str | `"softsign"` | See activation list below |
| `A` | float | 1.0 | Sinusoidal drive amplitude |
| `omega_drive` | float | 1.0 | Drive frequency |
| `phi` | array (n,) | zeros | Per-neuron phase offsets |
| `b` | array (n,) | zeros | Constant bias per neuron |
| `oja_frozen` | bool | False | If True, weights are held fixed |

### Activation functions

`tanh`, `softsign`, `arctan`, `centered_sigmoid`, `clipped_gelu`,
`bounded_gelu`, `sigmoid_firing`, `bounded_softplus`, `smooth_capped_relu`

### Sweep ranges (`PARAM_RANGES` dict in `fourier.py`)

```python
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
```

Override by passing `param_ranges=your_dict` to `run_fourier_experiment`.

---

## Output plots

| File | What it shows |
|---|---|
| `freq_matrix.png` | Median R² for every `(omega_drive, omega_target)` pair, split into learning vs frozen panels. Off-diagonal entries show frequency-transformation performance. |
| `activation_boxplot.png` | Distribution of R² for each activation function, learning vs frozen. |
| `neuron_count.png` | How R² changes with `n` (more neurons = richer basis). |
| `heatmap_tau_vs_omega_drive.png` | Interaction between neural time constant and drive frequency. |
| `heatmap_eta_vs_lambda.png` | Plasticity regime: fast vs slow learning / decay. |
| `heatmap_gain_vs_n.png` | Nonlinearity strength vs network size. |
| `best_reconstruction_static.png` | Target and best reconstruction overlaid, with residual. |
| `best_reconstruction.gif` | Sliding-window animation. Two rows: Oja learning (top) vs Oja frozen (bottom). Left column = full signal with window markers. Right column = zoomed window with local R² in title. |

---

## Extending

### Adding a new activation function

1. Add a branch in `activation()` in Section 1.
2. Append the name to `ALL_ACTIVATIONS`.

### Testing complex target waves

Replace the target in `fit_linear_readout` — change `np.sin(omega_target * t_ss)` to any waveform (e.g. a square wave, a sum of sines).

### Increasing neurons beyond 12

Add larger values to `PARAM_RANGES["n"]`. Runtime scales as O(n²) due to the weight matrix.

### Parallelising the sweep

`random_sweep` is embarrassingly parallel over the `i` loop. Wrap `run_experiment` calls in `concurrent.futures.ProcessPoolExecutor` for a straightforward speedup.

### Interactive display (non-headless)

If running in a Jupyter notebook or interactive terminal, comment out `matplotlib.use("Agg")` if present, or call `plt.show()` after each `savefig`.

---

## Legacy analysis

Section 9 of `fourier.py` preserves the original 3-neuron recurrence-domain analysis. Uncomment the loop at the bottom of the file to run it:

```python
for act in ALL_ACTIVATIONS:
    run_analysis_for_activation(act, T=300, dt=0.05)
```
