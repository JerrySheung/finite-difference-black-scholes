# Crank-Nicolson Finite Difference Solver for European Options

An efficient, robust numerical solver for the Black-Scholes partial differential equation (PDE) using the **Crank-Nicolson (implicit-explicit) finite difference method**. The solver values European call and put options and matches numerical grid results against the exact Black-Scholes analytical solution.

## Features
- **Crank-Nicolson Scheme:** Second-order accurate in both space and time ($O(\Delta t^2, \Delta S^2)$), offering unconditional stability compared to explicit Euler methods.
- **Sparse Matrix Optimization:** Uses `scipy.sparse.csr_matrix` for high-performance, memory-efficient tridiagonal system storage and scaling.
- **Dynamic Time-Dependent Boundaries:** Accurately injects time-decaying Dirichlet boundary conditions at each time step.
- **Validation Engine:** Automatically benchmarks grid outputs against the analytical Black-Scholes formula using `scipy.stats.norm`.

## Mathematical Overview
The Black-Scholes PDE is transformed via a theta-scheme where $\theta = 0.5$, averaging the implicit and explicit time steps:

$$ \frac{V^{n+1} - V^n}{\Delta t} + \frac{1}{2} \left( \mathcal{L} V^{n+1} + \mathcal{L} V^n \right) = 0 $$

Where $\mathcal{L}$ is the spatial differential operator. This results in solving a tridiagonal linear system $B \cdot V^{n} = A \cdot V^{n+1} + \text{Boundaries}$ backward in time from maturity $T \to 0$.

## Getting Started

### Prerequisites
Make sure you have the core scientific computing stack installed:
```bash
pip install numpy scipy matplotlib