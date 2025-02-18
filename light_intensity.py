import numpy as np  
import matplotlib.pyplot as plt
N = 100  # Number of depth points
L = 100  # Water column depth (m)
z = np.linspace(0, L, N)  # Depth grid
I_0 = 100  # Surface light intensity (W/m²)
k = 0.1  # Background light attenuation coefficient (m⁻¹)
alpha = 0.03  # Self-shading coefficient (m²/mg)

# Initialize phytoplankton concentration
P = np.exp(-((z - L/2) / (L/10))**2)


def calculate_light(P, z, I_0=100, k=0.1, alpha=0.03):
    """
    Compute light intensity at each depth based on phytoplankton concentration.
    """
    P_total = np.trapz(P, z)  # Integrate P over depth for self-shading
    I = I_0 * np.exp(-k * z) * np.exp(-alpha * P_total)  # Lambert-Beer Law
    return I

# Define different diffusion coefficients and water depths to match profiles in Huisman & Sommeijer (2002)
diffusion_values = [0.01, 0.1, 1.0]  # Low, medium, high diffusion
depth_values = [50, 100, 200]  # Different water depths

# Create figure for multiple profiles
fig, axes = plt.subplots(len(diffusion_values), len(depth_values), figsize=(12, 10), sharex=True, sharey=True)

# Iterate over different diffusion and depth values
for i, D in enumerate(diffusion_values):
    for j, L in enumerate(depth_values):
        # New depth grid for each L
        N = 100  # Keeping resolution fixed
        z = np.linspace(0, L, N)
        dz = L / N

        # Initial phytoplankton concentration (Gaussian patch)
        P = np.exp(-((z - L/2) / (L/10))**2)

        # Solve until convergence
        previous_P = np.zeros_like(P)
        for iteration in range(1000):  # Max iterations
            I_values = calculate_light(P, z, I_0, k, alpha)  # Compute light
            P_new = P + 0.01 * (I_values / (I_values + 20)) * P - 0.005 * P  # Growth-loss dynamics

            # Check for convergence
            if np.linalg.norm(P_new - previous_P) < 1e-6:
                break

            previous_P = P.copy()
            P = P_new.copy()

        # Plot steady-state phytoplankton profile
        axes[i, j].plot(P, -z, label=f"D={D}, L={L}")
        axes[i, j].set_title(f"D={D}, L={L}")
        axes[i, j].grid()

# Set axis labels
for ax in axes[-1, :]:
    ax.set_xlabel("Phytoplankton Concentration (mg/m³)")
for ax in axes[:, 0]:
    ax.set_ylabel("Depth (m)")

# Adjust layout and show plots
plt.suptitle("Phytoplankton Profiles with Different Diffusion & Water Depths")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
