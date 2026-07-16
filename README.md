# Crank-Nicolson Finite Difference Solver for European Options
A numerical solver for the Black-Scholes partial differential equation (PDE) using the Crank-Nicolson finite difference method. The solver values European call and put options and matches numerical grid results against the exact Black-Scholes analytical solution.

## Features
- **Crank-Nicolson Scheme:** Second-order accurate in both space and time, offering unconditional stability compared to explicit Euler methods.
- **Sparse Matrix Optimization:** Uses `scipy.sparse.csr_matrix` for high-performance, memory-efficient tridiagonal system storage and scaling.
- **Dynamic Time-Dependent Boundaries:** Uses Dirichlet boundary conditions at each time step.
- **Validation Engine:** Automatically benchmarks grid outputs against the analytical Black-Scholes formula.

## Mathematical Overview
The Black-Scholes PDE is transformed averaging the implicit and explicit time steps:

$$ \frac{V^{n+1} - V^n}{\Delta t} + \frac{1}{2} \left( \mathcal{L} V^{n+1} + \mathcal{L} V^n \right) = 0 $$

Where $\mathcal{L}$ is the spatial differential operator. This results in solving a tridiagonal linear system $B \cdot V^{n} = A \cdot V^{n+1} + g$, backward in time from maturity $T \to 0$.


## Packages used
Make sure you have these packages installed:
```bash
pip install numpy scipy matplotlib