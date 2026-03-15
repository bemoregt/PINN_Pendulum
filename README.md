# Pendulum PINN Simulator

A desktop GUI application that trains a **Physics-Informed Neural Network (PINN)** to learn the motion of a simple pendulum and visualises the result as an interactive animation.

Accuracy against a 4th-order Runge–Kutta reference: **< 0.01 °** mean absolute error over a 10 s trajectory.

---

## Physics

The simulator models a nonlinear pendulum governed by the second-order ODE:

```
θ''(t) + (g / l) · sin θ(t) = 0
```

| Symbol | Meaning |
|--------|---------|
| `θ(t)` | Angular displacement (rad) |
| `g`    | Gravitational acceleration (m/s²) |
| `l`    | Pendulum length (m) |

No small-angle approximation is made — the full nonlinear equation is enforced at every collocation point.

---

## What is a PINN?

A **Physics-Informed Neural Network** is a neural network whose loss function encodes the governing physics, so the learned solution automatically satisfies the ODE and the initial conditions.

### Loss function (per segment)

```
L_total = L_physics + γ · L_energy + λ · L_ic

L_physics = mean[ (θ'' + (g/l) sinθ)² ]          ODE residual at collocation points
L_energy  = mean[ (½ω² + (g/l)(1−cosθ) − E₀)² ]  energy conservation
L_ic      = (θ(t_s) − θ_ic)² + (ω(t_s) − ω_ic)²  initial / continuity conditions
```

`θ'` and `θ''` are obtained via **automatic differentiation** through the network (`torch.autograd.grad`) — no finite-difference approximation is used.

`L_energy` explicitly penalises deviations from the conserved energy `E₀ = ½ω₀² + (g/l)(1−cosθ₀)`, which prevents the network from collapsing to the trivial equilibrium `θ = 0`.

---

## Architecture: SIREN + Time-Marching

### Why SIREN?

Standard Tanh MLPs suffer from **spectral bias**: they learn low-frequency components first and cannot represent oscillatory solutions accurately enough for the ODE residual to reach near-zero.
**SIREN** (Sinusoidal Representation Networks, Sitzmann et al. 2020) uses `sin(ω₀ · Wx + b)` activations.  Because every derivative of a sinusoid is also a sinusoid, the automatic-differentiation graph stays well-conditioned and the physics residual drops to `~1e-4` — roughly **1 000×** lower than the Tanh baseline.

| Activation | Physics loss (1 segment, 3000 epochs) | Full-trajectory MAE |
|------------|--------------------------------------|---------------------|
| Tanh MLP   | ~0.22 (stagnates)                    | ~15 °               |
| **SIREN**  | **~1e-4**                            | **< 0.01 °**        |

SIREN weight initialisation (Sitzmann et al. 2020):

```
first layer  : W ~ U(−1/n_in,  +1/n_in)
hidden layers: W ~ U(−√(6/n_in)/ω₀,  +√(6/n_in)/ω₀)
```

### Why Time-Marching?

A single PINN over a long time horizon (e.g. 10 s ≈ 5 oscillations) faces an exponentially complex loss landscape and invariably converges to a wrong local minimum.
**Time-marching** (shooting method) breaks the problem into `n_seg` short segments, each covering roughly one oscillation:

```
[0, T/n] → [T/n, 2T/n] → … → [(n-1)T/n, T]
```

Each segment trains an independent SIREN on its sub-domain.  After training, the endpoint `(θ, ω)` is extracted via autograd and passed as the initial condition to the next segment.  Each sub-problem is easy: one oscillation, well-conditioned, fast convergence.

```
Segment k
──────────────────────────────────────────────────
  Input : t ∈ [t_s, t_e]   (normalised to [−1, 1])
  Train : L_physics + γ L_energy + λ L_ic
  Output: θ(t)  for t ∈ [t_s, t_e]
  Propagate: θ(t_e), ω(t_e)  →  IC for segment k+1
```

---

## Features

- **Configurable parameters** — `g`, `l`, `θ₀`, `ω₀`, `T`, epochs per segment, number of segments, collocation points, IC weight `λ`, energy weight `γ`.
- **Background training** — each segment trains in a worker thread; the GUI stays responsive throughout.
- **Live loss curve** — log-scale loss updates in real time, showing per-segment physics / energy / IC breakdown in the status label.
- **Animated pendulum** — PINN prediction drives a smooth pendulum animation with a fading bob trail.
- **Comparison plots** — `θ(t)` and phase portrait `(θ, ω)` overlay the PINN result with the RK4 reference; vertical dotted lines mark segment boundaries.
- **Synchronised cursor** — a green marker on the `θ(t)` plot tracks the animated pendulum in real time.

---

## GUI Layout

```
┌──────────────────────┬──────────────────────────────────────────┐
│  Parameters          │                                          │
│  g, l, θ₀, ω₀, T    │        Pendulum Animation                │
│  epochs/seg, n_seg   │          (PINN prediction)               │
│  n_col, λ, γ         │                                          │
│                      ├────────────────────┬─────────────────────┤
│  Training progress   │   θ(t) comparison  │   Phase portrait    │
│  [progress bar]      │   PINN vs RK4      │   PINN vs RK4       │
│  Loss: …             │   (segment marks)  │                     │
│  [Train PINN]        │                    │                     │
│  [Animate]           │                    │                     │
│  [Stop]              │                    │                     │
│  Loss curve (log)    │                    │                     │
└──────────────────────┴────────────────────┴─────────────────────┘
```

---

## Requirements

| Package    | Version tested |
|------------|----------------|
| Python     | ≥ 3.9          |
| PyTorch    | ≥ 2.0          |
| NumPy      | ≥ 1.24         |
| Matplotlib | ≥ 3.7          |
| tkinter    | stdlib (bundled with CPython) |

Install dependencies:

```bash
pip install torch numpy matplotlib
```

> On macOS, `tkinter` requires a Python built with Tcl/Tk support:
> ```bash
> brew install python-tk
> ```

---

## Usage

```bash
python pinn_pendulum_gui.py
```

**Typical workflow:**

1. Adjust parameters in the left panel (or keep the defaults).
2. Click **Train PINN** — the progress bar advances segment by segment; watch each segment's physics / energy / IC loss in the status label.
3. Click **Animate** once training completes to see the pendulum swing and compare PINN vs RK4.
4. Click **Stop** to pause the animation, then re-train with different parameters if desired.

**Recommended defaults** (good balance of speed and accuracy):

| Parameter | Default | Notes |
|-----------|---------|-------|
| Epochs / seg | 3000 | 1200+ usually sufficient for simple cases |
| Time segments | 5 | Increase for large `T` or large `θ₀` |
| Colloc. points / seg | 200 | 100–300 works well |
| IC weight λ | 500 | Higher = stricter continuity across segments |
| Energy weight γ | 1.0 | Prevents θ→0 trivial solution |

---

## Code Structure

```
pinn_pendulum_gui.py
│
├── SegmentPINN          SIREN network for one time segment
│     └── forward()      normalises t to [−1,1], applies sin(ω₀·Wx+b)
│
├── segment_loss()       ODE residual + energy conservation + IC
│
├── _theta_at()          evaluate θ at a single time point (no grad)
├── _omega_at()          evaluate ω = dθ/dt via autograd
│
├── rk4_pendulum()       4th-order Runge–Kutta reference solver
│
└── PendulumApp          tkinter GUI + training orchestration
      ├── _train_worker()  time-marching loop (background thread)
      ├── _predict()       stitch segment predictions
      └── _start_animation()  FuncAnimation with synchronised cursor
```

### Numerical Reference

A **4th-order Runge–Kutta** (RK4) integrator with 2 000 steps provides a high-accuracy reference trajectory used for visual comparison only — it is **not** used during PINN training.

---

## Design Decisions & Lessons Learned

| Approach tried | Why it failed |
|----------------|---------------|
| Single Tanh MLP over full T | Spectral bias; physics loss stagnates ~0.22; θ→0 trivial solution |
| Fourier features + causal weighting + L-BFGS | High-frequency features → gradient explosion → NaN |
| Curriculum expansion + energy loss (Tanh) | Tanh still cannot satisfy ODE residual; MAE ~15° |
| **Time-marching + SIREN** ✓ | Each segment ≈ 1 oscillation; SIREN physics loss ~1e-4; MAE < 0.01° |

---

## License

MIT
