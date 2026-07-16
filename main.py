import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
import matplotlib.pyplot as plt


def solve_bs(S_max, K, T, r, sigma, option, M = 500, N = 1000):

    if option not in ['call', 'put']:
        raise ValueError("Option must be 'call' or 'put'")

    # Time and space step sizes
    dt = T / N
    dS = S_max / M

    # Initialse grid
    S = np.linspace(0, S_max, M + 1)
    i = np.arange(1, M)


    # Matrix construction
    a = (dt / 2) * (1 / 2) * (sigma**2 * i**2 - r * i)
    b = (dt / 2) * (- 1) * (sigma**2 * i**2 + r)
    c = (dt / 2) * (1 / 2)*(sigma**2 * i**2 + r * i)

    A_lower_diag = a[1:]
    A_diag = 1 + b
    A_upper_diag = c[:-1]
    A = sparse.diags([A_lower_diag, A_diag, A_upper_diag], offsets = [-1, 0, 1], format = 'csr')

    B_lower_diag = -a[1:]
    B_diag = 1 - b
    B_upper_diag = -c[:-1]
    B = sparse.diags([B_lower_diag, B_diag, B_upper_diag], offsets = [-1, 0, 1], format = 'csr')


    # Initial condition at maturity
    if option == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)


    # Price backwards in time
    for t_step in range(N - 1, -1, -1):
        
        # current time
        t = t_step * dt
        # time to maturity
        tau = T - t

        explicit = A @ V[1:M]

        if option == 'call':
            V_0_current, V_0_future = 0, 0
            V_M_current, V_M_future = S_max - K * np.exp(-r * tau), S_max - K * np.exp(-r * (tau - dt))
        else:
            V_0_current, V_0_future = K * np.exp(-r * tau), K * np.exp(-r * (tau - dt))
            V_M_current, V_M_future = 0, 0

        # Boundary condition adjustment
        explicit[0] += a[0] * (V_0_future + V_0_current)
        explicit[-1] += c[-1] * (V_M_future + V_M_current)

        # Solve the system of equations
        V[1:M] = spsolve(B, explicit)
        V[0] = V_0_current
        V[-1] = V_M_current
        
    return S, V


def black_scholes(S0, K, T, r, sigma, option):
    if option not in ['call', 'put']:
        raise ValueError("Option must be 'call' or 'put'")
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    


def main():
    S_max = 300.0    
    K = 100.0        
    T = 1.0          
    r = 0.05         
    sigma = 0.20     
    option_type = 'put'
    
    # Numerical solutions
    S_grid, V_numerical = solve_bs(S_max, K, T, r, sigma, option_type, M= 500, N=1000)

    # Exact heoretical solutions
    V_exact = np.zeros_like(S_grid)
    for idx, s in enumerate(S_grid):
        if s > 0:
            V_exact[idx] = black_scholes(s, K, T, r, sigma, option_type)
        else:
            V_exact[idx] = 0 if option_type == 'call' else K * np.exp(-r * T)

    # Payoff at maturity
    V_payoff = np.maximum(S_grid - K, 0) if option_type == 'call' else np.maximum(K - S_grid, 0)

    # Plotting
    plt.plot(S_grid, V_payoff, color = 'gray', linestyle = '--', linewidth = 1.5, label = 'Payoff at Expiration (t=T)')
    plt.plot(S_grid, V_exact, color = 'blue', linestyle = '-', linewidth = 2, label = 'Exact Black-Scholes (t=0)')
    # Plotting every 4 points
    plt.scatter(S_grid[::4], V_numerical[::4], color = 'red', marker = 'o', s = 15, facecolors = 'none', label = 'Crank-Nicolson Grid')

    plt.xlim(K * 0.4, K * 1.8) 
    plt.ylim(-2, K * 0.9)
    plt.title(f'European {option_type.capitalize()} Price: Crank-Nicolson vs. Analytical Black-Scholes', fontsize=12, fontweight='bold')
    plt.xlabel('Underlying Asset Price ($S$)', fontsize=10)
    plt.ylabel('Option Value ($V$)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)

    plt.show()


if __name__ == "__main__":
    main()