import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Step 1: Define Parameters
N_values = [50, 100, 200, 400]  # Different resolutions to test convergence
L = 100.0  # Total depth (meters)
u = 0.01  # Sinking velocity (m/s)
D = 0.1  # Diffusion coefficient (m²/s)
growth_rate = 0.015  # Phytoplankton growth rate
loss_rate = 0.003  # Phytoplankton loss rate
K = 1.0  # Carrying capacity (limits growth)
T_max = 1000  # Simulation time (seconds)
t_eval = np.linspace(0, T_max, 100)  # Time points

# Define the Flux Equations
def compute_fluxes(P, dz, u, D):
    J_adv = u * np.roll(P, 1)  # Upwind advection
    J_diff = -D * (np.roll(P, -1) - P) / dz  # Central difference diffusion
    J = J_adv + J_diff
    J[0] = J[-1] = 0  # No flux boundary condition
    return J

# Define the Differential Equation for ODE Solver
def dPdt(t, P, N, dz, u, D, growth_rate, loss_rate, K):
    J = compute_fluxes(P, dz, u, D)
    dP = np.zeros_like(P)
    dP[1:-1] = -(J[1:-1] - J[0:-2]) / dz  # Finite volume method
    dP += growth_rate * P * (1 - P / K) - loss_rate * P  # Logistic growth with loss
    return dP

# Solve for Different Resolutions and Store Data
convergence_results = {}
depths = {}
for N in N_values:
    dz = L / N  # Grid spacing
    z = np.linspace(0, L, N)  # Depth grid
    P0 = np.exp(-((z - L/2) / (L/10))**2)  # Initial condition

    solution = solve_ivp(dPdt, [0, T_max], P0, args=(N, dz, u, D, growth_rate, loss_rate, K), 
                         method='RK45', t_eval=t_eval)

    convergence_results[N] = solution.y  # Store full time evolution
    depths[N] = z

# Plot Convergence Results with Colormap
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, N in zip(axes.flatten(), N_values):
    im = ax.imshow(convergence_results[N], aspect='auto', origin='lower',
                   extent=[0, T_max, 0, L], cmap='plasma')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"Phytoplankton Concentration (N={N})")

fig.colorbar(im, ax=axes, orientation='vertical', label="Phytoplankton Concentration")
plt.suptitle("Convergence Test: Effect of Grid Resolution on Solution")
plt.show()
