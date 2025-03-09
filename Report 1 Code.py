import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#
# Define Parameters Class
#
class Parameters:
    def __init__(self, D=5, u=1, depth=300, dz=10,
                 L0=400, Lamplitude=0, T=365,
                 mortality=0.1, N_bottom=350, mu_max=1,
                 K_N=0.0425, light_attenuation=6e-10,
                 alphaL=0.1, water_attenuation=0.045, eff=0.5):

        self.D = D  # Diffusivity (m²/day)
        self.u = u  # Settling velocity (m/day)
        self.depth = depth  # Depth (meters)
        self.dz = dz  # Grid spacing (meters)
        self.L0 = L0  # Surface light intensity (W/m²)
        self.Lamplitude = Lamplitude  # Seasonal amplitude fraction
        self.T = T  # Period of seasonal variation (days)
        self.mortality = mortality  # Phytoplankton mortality rate (1/day)
        self.N_bottom = N_bottom  # Bottom nutrient concentration (mmol/m³)
        self.mu_max = mu_max  # Maximum growth rate (1/day)
        self.K_N = K_N  # Half-saturation constant for nitrogen uptake (mmol/m³)
        self.light_attenuation = light_attenuation  # Light damping by phytoplankton (m²/cell)
        self.alphaL = alphaL  # Affinity for light
        self.water_attenuation = water_attenuation  # Light damping by water (1/m)
        self.eff = eff  # Efficiency of nutrient recycling

#
# Seasonal Light Variation Function
#
def seasonal_light(t, P, param, z):
    """Seasonal variation in surface light intensity including water and phytoplankton shading."""
    I_surface = param.L0 * (1 + param.Lamplitude * np.sin(2 * np.pi * t / param.T))
    return I_surface * np.exp(-param.water_attenuation * z - param.light_attenuation * np.cumsum(P) * param.dz)

#
# Nutrient Uptake Function (Michaelis-Menten Type)
#
def nutrient_uptake(N, param):
    """Nutrient uptake function based on Michaelis-Menten kinetics."""
    return param.mu_max * (N / (param.K_N + N))

#
# Advection-Diffusion Model with Nutrients and Seasonal Light
#
def pmodel_deriv_full(t, state, param):
    """Advection-diffusion-reaction model with seasonal light, nutrients, and stability."""

    # Extract state variables (P = Phytoplankton, N = Nutrients)
    nGrid = param.nGrid
    P = state[:nGrid]
    N = state[nGrid:]

    # Ensure non-negative values
    P = np.maximum(P, 1e-6)
    N = np.maximum(N, 1e-6)

    # Seasonal light intensity at depth
    I_z = seasonal_light(t, P, param, np.linspace(0, param.depth, nGrid))

    # Functional response for light
    sigma_I = param.alphaL * I_z / (param.alphaL * I_z + param.mu_max)

    # Growth rate considering nutrient & light limitation
    growth = nutrient_uptake(N, param) * sigma_I

    # Advective fluxes for phytoplankton
    Jadv_P = np.zeros(nGrid + 1)
    Jadv_P[1:nGrid] = param.u * P[:nGrid - 1]
    Jadv_P[nGrid] = param.u * P[-1]  # Open bottom

    # No advective flux for nutrients

    # Diffusive fluxes (Central difference)
    Jdiff_P = np.zeros(nGrid + 1)
    Jdiff_P[1:nGrid] = -param.D * (P[1:nGrid] - P[:nGrid - 1]) / param.dz

    Jdiff_N = np.zeros(nGrid + 1)
    Jdiff_N[1:nGrid] = -param.D * (N[1:nGrid] - N[:nGrid - 1]) / param.dz

    # Boundary condition for nutrient diffusion
    Jdiff_N[nGrid] = -param.D * (param.N_bottom - N[-1]) / param.dz

    # Compute net flux for phytoplankton
    dPdt = -(Jadv_P[1:nGrid + 1] - Jadv_P[:nGrid]) / param.dz \
           -(Jdiff_P[1:nGrid + 1] - Jdiff_P[:nGrid]) / param.dz \
           - param.mortality * P + growth * P

    # Compute net flux for nutrients
    dNdt = -(Jdiff_N[1:nGrid + 1] - Jdiff_N[:nGrid]) / param.dz \
           + param.eff * param.mortality * P - growth * P

    return np.concatenate([dPdt, dNdt])  # Return both phytoplankton and nutrients

#
# Solve the Model with RK45 Solver
#
def advection_diffusion_full_rk45(param):
    z = np.linspace(0, param.depth, int(param.depth / param.dz) + 1)
    param.nGrid = len(z)  # No. of grid cells

    # Initialization: Gaussian phytoplankton concentration profile
    P0 = np.exp(-((z - param.depth / 2) ** 2) / 50)
    N0 = np.ones_like(P0) * param.N_bottom

    state0 = np.concatenate([P0, N0])

    # Solve model for 100 days with `RK45` solver
    sol = solve_ivp(pmodel_deriv_full, [0, 100], state0, args=(param,), method='RK45')

    return sol.t, z, sol.y


# Run final analysis
param = Parameters()
t, z, y = advection_diffusion_full_rk45(param)

# Separate phytoplankton and nutrients
P = y[:param.nGrid, :]
N = y[param.nGrid:, :]

# Plot Phytoplankton and Nutrients over Time
plt.figure(figsize=(8, 6))

# Phytoplankton Plot
plt.subplot(2, 1, 1)
plt.pcolormesh(t, -z, P, shading='auto', cmap='viridis')
plt.ylabel('Depth (m)')
plt.colorbar(label='Phytoplankton (mmol N/m³)')
plt.title('Phytoplankton Concentration Over Time')

# Nutrients Plot
plt.subplot(2, 1, 2)
plt.pcolormesh(t, -z, N, shading='auto', cmap='plasma')
plt.ylabel('Depth (m)')
plt.xlabel('Time (days)')
plt.colorbar(label='Nutrients (mmol N/m³)')
plt.title('Nutrient Concentration Over Time')

plt.tight_layout()
plt.show()

# Function to Plot Phytoplankton Profiles
#
def plot_phytoplankton_profiles_final(z, P_list, title, param_values, param_name, colors, depth):
    plt.figure(figsize=(8, 6))

    for i, param_value in enumerate(param_values):
        P_final = P_list[i][:, -1]  # Extract final time step of phytoplankton concentration
#
# Interpolation for smooth curves
        z_smooth = np.linspace(z.min(), z.max(), 300)
        P_smooth = interp1d(z, P_final, kind='cubic', fill_value='extrapolate')(z_smooth)

        plt.plot(P_smooth, z_smooth, label=f"{param_name} = {param_value}", color=colors[i], linewidth=2)

    plt.xlabel("Phytoplankton Concentration (mmol N m⁻³)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.legend(title=f"{param_name} ({'1/day' if 'Mortality' in param_name else 'W/m²'})")

    # Fix depth axis: Surface (0m) at the top, Deep (300m) at the bottom
    plt.ylim(depth, 0)  

    # Auto-adjust x-axis for proper visualization
    P_flat = np.array([P_val[:, -1] for P_val in P_list]).flatten()
    plt.xlim(0, np.max(P_flat) * 1.1)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Run Sensitivity Analysis and Ensure Plots are Displayed
#
def sensitivity_analysis_final():
    mortality_values = [0.1, 0.2]  # Two mortality values
    light_values = [200, 400]  # Two light intensities
    depth = 300  # Depth of the system

    P_mortality = []
    P_light = []

    # Vary mortality rate
    for mortality in mortality_values:
        param = Parameters(mortality=mortality)
        t, z, y = advection_diffusion_full_rk45(param)  # Correct function name
        P = y[:param.nGrid, :]  # Extract phytoplankton
        P_mortality.append(P)

    # Vary surface light
    for L0 in light_values:
        param = Parameters(L0=L0)
        t, z, y = advection_diffusion_full_rk45(param)  # Correct function name
        P = y[:param.nGrid, :]  # Extract phytoplankton
        P_light.append(P)

    # Ensure valid data before plotting
    if (P_mortality):
        plot_phytoplankton_profiles_final(z, P_mortality,
                                          "Effect of Mortality on Phytoplankton Profiles (300m Depth)",
                                          mortality_values, "Mortality Rate",
                                          ["blue", "red"], depth)

    if (P_light):
        plot_phytoplankton_profiles_final(z, P_light,
                                          "Effect of Light on Phytoplankton Profiles (300m Depth)",
                                          light_values, "Light Intensity",
                                          ["purple", "yellow"], depth)

# Run the final sensitivity analysis to ensure plots appear
sensitivity_analysis_final()