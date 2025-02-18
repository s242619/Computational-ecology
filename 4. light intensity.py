import numpy as np

def calculate_light(P, z, I_0=100, k=0.1, alpha=0.03):
    """
    Calculate light intensity at each depth based on phytoplankton concentration.

    Parameters:
    P (array): Phytoplankton concentration at different depths (mg/m³).
    z (array): Depth levels (m).
    I_0 (float): Surface light intensity (W/m²) (default: 100).
    k (float): Background light attenuation coefficient (m⁻¹) (default: 0.1).
    alpha (float): Self-shading coefficient (m²/mg) (default: 0.03).

    Returns:
    I (array): Light intensity at each depth (W/m²).
    """
    P_total = np.trapz(P, z)  # Integrate P(z) over depth to get total biomass
    I = I_0 * np.exp(-k * z) * np.exp(-alpha * P_total)  # Apply Lambert-Beer’s Law
    return I

# Example Usage
z = np.linspace(0, 100, 100)  # Depths from 0 to 100 meters
P = np.exp(-((z - 50) / 10) ** 2)  # Example phytoplankton patch (Gaussian)

I_values = calculate_light(P, z)

# Plot the result
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.plot(I_values, -z, label="Light Intensity")
plt.xlabel("Light Intensity (W/m²)")
plt.ylabel("Depth (m)")
plt.title("Light Intensity vs Depth")
plt.legend()
plt.grid()
plt.show()
 