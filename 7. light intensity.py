#Compute light intensity with and without self-shading
# Define the phytoplankton concentration P (Gaussian patch in the middle)
import numpy as np  # Ensure NumPy is imported
import matplotlib.pyplot as plt

N = 100  # Number of depth points
L = 100  # Water column depth (m)
z = np.linspace(0, L, N)  # Depth grid
I_0 = 100  # Surface light intensity (W/m²)
k = 0.1  # Background light attenuation coefficient (m⁻¹)
alpha = 0.03  # Self-shading coefficient (m²/mg)

# Initialize phytoplankton concentration
P = np.exp(-((z - L/2) / (L/10))**2)


# Define the calculate_light function
def calculate_light(P, z, I_0=100, k=0.1, alpha=0.03):
    """
    Compute light intensity at each depth based on phytoplankton concentration.
    """
    P_total = np.trapz(P, z)  # Use np.trapz instead of np.trapezoid
    I = I_0 * np.exp(-k * z) * np.exp(-alpha * P_total)  # Lambert-Beer Law
    return I   
I_with_shading = calculate_light(P, z, I_0, k, alpha)  # Normal case
I_without_shading = calculate_light(P, z, I_0, k, 0)  # No self-shading (alpha = 0)

# Create a figure with three subplots
fig, axes = plt.subplots(3, 1, figsize=(6, 12))

# Plot phytoplankton concentration
axes[0].plot(P, -z, label="Steady-State Phytoplankton")
axes[0].set_xlabel("Phytoplankton Concentration (mg/m³)")
axes[0].set_ylabel("Depth (m)")
axes[0].set_title("Phytoplankton Concentration vs. Depth")
axes[0].legend()
axes[0].grid()

# Plot light intensity with self-shading
axes[1].plot(I_with_shading, -z, label="Light with Self-Shading", color="orange")
axes[1].set_xlabel("Light Intensity (W/m²)")
axes[1].set_ylabel("Depth (m)")
axes[1].set_title("Light Intensity with Self-Shading")
axes[1].legend()
axes[1].grid()

# Plot light intensity with and without self-shading
axes[2].plot(I_with_shading, -z, label="With Self-Shading", color="orange")
axes[2].plot(I_without_shading, -z, label="Without Self-Shading", linestyle="dashed", color="blue")
axes[2].set_xlabel("Light Intensity (W/m²)")
axes[2].set_ylabel("Depth (m)")
axes[2].set_title("Comparison: Light With vs. Without Self-Shading")
axes[2].legend()
axes[2].grid()

# Adjust layout and show plots
plt.tight_layout()
plt.show()
