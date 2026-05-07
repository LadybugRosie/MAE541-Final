# Chaos / Stability Experiment — Map vs. Flow RNNs

## Motivation

Funamizu & Karakida (2025) state explicitly (Section 2) that random RNNs undergo an
**order-to-chaos phase transition** as the spectral radius of the weight matrix W crosses a
critical value.  The two standard formulations of an RNN give different routes to that
transition:

| Formulation | Equation | Type |
|---|---|---|
| **Discrete map** | h_{t+1} = W f(h_t) + W_in x_t | iterated map |
| **Continuous flow** | τ ḣ(t) = −h(t) + W f(h(t)) + W_in x(t) | autonomous ODE |

The continuous flow is the infinitesimal-time limit of the map (obtained by subtracting
h_t from both sides and letting dt → 0).  Despite this close relationship, the two
formulations generate qualitatively different transition routes to chaos:

- **Discrete map** → period-doubling (flip bifurcation) cascade → chaos
- **Continuous flow** → Hopf bifurcation → limit cycle → torus → chaos (Ruelle–Takens)

Both share the same critical spectral-radius threshold near the origin, but differ in the
geometry and speed of the transition.

---

## Theoretical Background

### Random-matrix initialisation

Weight entries drawn as W_{ij} ~ N(0, g²/N) give spectral radius ρ(W) ≈ g by the
circular law of random matrix theory (Sompolinsky et al. 1988).

| Regime | g | Behaviour |
|---|---|---|
| Ordered / stable | g < 1 | fixed-point attractor |
| Edge of chaos | g ≈ 1 | critical, longest memory |
| Chaotic | g > 1 | positive Lyapunov exponent |

### Linearisation

Near the zero fixed point, with f(0)=0 and f′(0)=1 (e.g. tanh):

- **Discrete map Jacobian**: J = W diag(f′(h)) → W at h=0. Stable iff ρ(W) < 1.
- **Continuous flow Jacobian**: J = (−I + W diag(f′(h)))/τ → (−I + W)/τ at h=0.
  Stable iff all eigenvalues of W have real part < 1.

For symmetric activation families the thresholds coincide at ρ(W) = 1, but the
**rate** at which chaos develops and the **dimension** of the resulting attractor differ.

### Largest Lyapunov exponent (LLE)

The LLE λ₁ distinguishes regimes cleanly:
- λ₁ < 0 → exponential contraction (stable)
- λ₁ = 0 → marginal (limit cycle / edge of chaos)
- λ₁ > 0 → chaos (exponential divergence of nearby trajectories)

Computed via the standard QR (Gram–Schmidt) method: propagate a small orthonormal
perturbation basis alongside the trajectory, re-orthonormalise at each step, and
accumulate the log of the expansion factors.

---

## Experiment Plan

### Phase 1 — Implement both formulations

**`rnn_models.py`**

```python
# Discrete map
def step_discrete(h, W, f, x=None, W_in=None):
    inp = W @ f(h)
    if x is not None and W_in is not None:
        inp += W_in @ x
    return inp   # h_{t+1}

# Continuous flow  (RHS of ODE for solve_ivp)
def rhs_continuous(t, h, W, f, tau, x=None, W_in=None):
    drive = W @ f(h) + (W_in @ x if x is not None else 0)
    return (-h + drive) / tau
```

Both share the same activation functions from `fourier/fourier.py` (import, do not copy).

**Initialisation**

```python
def init_W(n, g, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, g / np.sqrt(n), (n, n))
```

This gives ρ(W) ≈ g for large n.

---

### Phase 2 — Chaos metrics

**`chaos_metrics.py`**

1. **Largest Lyapunov exponent (QR method)**

   For the discrete map, evolve the tangent map:
   ```
   Q_{t+1} R_{t+1} = J_t Q_t,   J_t = W diag(f′(h_t))
   λ₁ = lim_{T→∞} (1/T) Σ log R_{ii}
   ```
   For the continuous flow, use the same scheme but integrate the variational
   equation alongside the ODE.

2. **Trajectory divergence** — two initial conditions separated by ε=1e-6;
   track δ(t) = ‖h₁(t) − h₂(t)‖. In the chaotic regime δ(t) ~ exp(λ₁ t).

3. **Autocorrelation decay** — C(τ) = ⟨xᵢ(t) xᵢ(t+τ)⟩. Decays to zero in the
   chaotic regime; remains elevated in the ordered regime.

4. **Power spectral density** — broadband in chaos, peaked in the limit-cycle regime.

---

### Phase 3 — Sweeps

**Sweep A: spectral radius g** (primary sweep)

```
g ∈ [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0]
```
For each g, compute LLE for both map and flow.  Repeat over multiple random seeds
(e.g. 10) and report mean ± std.

**Sweep B: activation function**

Test each of the 9 activation functions from `fourier.py` at the same g values.
Activation slope at the origin shifts the effective critical g:  f′(0) < 1 pushes
the threshold to g·f′(0) = 1, i.e. g_crit = 1/f′(0).

| Activation | f′(0) | Expected g_crit |
|---|---|---|
| tanh | 1.0 | 1.0 |
| softsign | 1.0 | 1.0 |
| arctan | 2/π ≈ 0.64 | ≈ 1.57 |
| centered_sigmoid | 0.5 | ≈ 2.0 |
| sigmoid_firing | 0.25 | ≈ 4.0 |
| bounded_softplus | ~0.5 | ≈ 2.0 |
| smooth_capped_relu | ~1.0 | ≈ 1.0 |
| clipped_gelu | ~1.0 | ≈ 1.0 |
| bounded_gelu | ~1.0 | ≈ 1.0 |

(exact values depend on gain parameter; these assume gain=1)

**Sweep C: network size n**

```
n ∈ [10, 50, 100, 500]
```
Tests whether the edge of chaos sharpens with n as expected from random matrix theory.

**Sweep D: time constant τ** (continuous flow only)

```
τ ∈ [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
```
τ controls how fast the flow responds. Very small τ → fast dynamics, easier to
go chaotic. Very large τ → sluggish; the −h/τ damping term dominates and
suppresses chaos. Predicts a τ-dependent shift of the effective critical g.

---

### Phase 4 — Visualisations

1. **LLE vs g** — line plot with both map and flow on the same axes; vertical
   dashed line at g=1 (theoretical threshold). Error bars from repeated seeds.

2. **Phase diagram** — 2D heatmap of LLE in (g, activation) space, separately
   for map and flow.

3. **Trajectory divergence** — log-linear plot of δ(t) vs t for g=0.5, 1.0, 1.5
   in both formulations side-by-side.

4. **Autocorrelation comparison** — for three g values (ordered / critical / chaotic),
   show C(τ) decay for both formulations.

5. **Power spectral density** — same three regimes, both formulations.

6. **n-dependence** — LLE vs g for n=10,50,100,500; expect the transition to
   sharpen into a step function as n → ∞.

7. **τ-sweep** (continuous only) — LLE vs g for each τ; expect the critical
   g to decrease as τ decreases.

---

## Proposed File Structure

```
chaos_stability/
├── rnn_models.py       # discrete step + continuous RHS, init_W
├── chaos_metrics.py    # LLE (QR), trajectory divergence, autocorr, PSD
├── sweep.py            # all four sweeps, returns DataFrames
├── visualize.py        # all seven plot types
├── run_experiment.py   # entry point: calls sweeps then saves all plots
└── README.md           # this file
```

Entry point:

```bash
uv run python chaos_stability/run_experiment.py
```

---

## Key Questions the Experiment Answers

| Question | Method |
|---|---|
| Does the map transition to chaos at the same g as the flow? | Sweep A, LLE vs g |
| Does the transition route differ (period-doubling vs Hopf)? | Trajectory plots + PSD |
| Which activation functions tolerate the highest g before going chaotic? | Sweep B |
| Does increasing n sharpen the phase boundary? | Sweep C |
| How does τ modulate the chaos threshold in the flow? | Sweep D |
| Is the edge of chaos (λ₁ ≈ 0) achievable in both formulations? | Sweep A, marker at λ₁=0 |
