import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Step 1: Define Parameters
N = 100          # Number of vertical grid points
L = 100.0        # Total depth (meters)
dz = L / N       # Grid spacing
u = 0.01         # Sinking velocity (m/s)
D = 0.1          # Diffusion coefficient (m²/s)
growth_rate = 0.015  # Phytoplankton growth rate
loss_rate = 0.003    # Phytoplankton loss rate
K = 1.0          # Carrying capacity (limits growth)
T_max = 1000     # Simulation time (seconds)

# Step 2: Initial Condition (Gaussian phytoplankton patch)
z = np.linspace(0, L, N)
P0 = np.exp(-((z - L/2) / (L/10))**2)  # Initial patch in the middle

# Step 3: Define the Flux Equations
def compute_fluxes(P, dz, u, D):
    J_adv = u * np.roll(P, 1)  # Upwind advection
    J_diff = -D * (np.roll(P, -1) - P) / dz  # Central difference diffusion
    J = J_adv + J_diff
    J[0] = J[-1] = 0  # No flux boundary condition
    return J

# Step 4: Define the Differential Equation for ODE Solver
def dPdt(t, P, N, dz, u, D, growth_rate, loss_rate, K):
    J = compute_fluxes(P, dz, u, D)
    dP = np.zeros_like(P)
    dP[1:-1] = -(J[1:-1] - J[0:-2]) / dz  # Finite volume method
    dP += growth_rate * P * (1 - P / K) - loss_rate * P  # Logistic growth with loss
    return dP

# Step 5: Solve the ODE Using solve_ivp
solution = solve_ivp(dPdt, [0, T_max], P0, args=(N, dz, u, D, growth_rate, loss_rate, K), 
                     method='RK45', t_eval=np.linspace(0, T_max, 100))

# Extract results
P_history = solution.y  # Phytoplankton concentration over time
time = solution.t

# Step 6: Plot the Results (Heatmap)
plt.figure(figsize=(8, 6))
plt.imshow(P_history, aspect='auto', origin='lower', extent=[0, T_max, 0, L], cmap='plasma')
plt.colorbar(label="Phytoplankton Concentration")
plt.xlabel("Time (seconds)")
plt.ylabel("Depth (m)")
plt.title("Phytoplankton Concentration Over Time")
plt.show()







