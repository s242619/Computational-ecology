# Solve the phytoplankton-light model until convergence
import numpy as np
import matplotlib.pyplot as plt  # Import for plotting

# Define light intensity function (needed in the script)
def calculate_light(P, z, I_0=100, k=0.1, alpha=0.03):
    """
    Compute light intensity at each depth based on phytoplankton concentration.
    """
    P_total = np.trapz(P, z)  # Integrate P over depth for self-shading
    I = I_0 * np.exp(-k * z) * np.exp(-alpha * P_total)  # Lambert-Beer Law
    return I

# Define parameters
I_0 = 100  # Surface light intensity (W/m²)
k = 0.1  # Background light attenuation coefficient (m⁻¹)
alpha = 0.03  # Self-shading coefficient (m²/mg)

# Spatial grid
N = 100  # Number of depth points
L = 100  # Water column depth (m)
z = np.linspace(0, L, N)  # Depth grid

# Initial phytoplankton concentration (Gaussian patch in the middle)
P = np.exp(-((z - L/2) / (L/10))**2)

# Define light intensity function
def calculate_light(P, z, I_0=100, k=0.1, alpha=0.03):
    """
    Compute light intensity at each depth based on phytoplankton concentration.
    """
    P_total = np.trapz(P, z)  # Integrate P over depth for self-shading
    I = I_0 * np.exp(-k * z) * np.exp(-alpha * P_total)  # Lambert-Beer Law
    return I

# Convergence criteria
tolerance = 1e-6
max_iterations = 1000
previous_P = np.zeros_like(P)

for iteration in range(max_iterations):
    # Compute light intensity at each depth
    I_values = calculate_light(P, z, I_0, k, alpha)
    
    # Update phytoplankton concentration based on light availability
    P_new = P + 0.01 * (I_values / (I_values + 20)) * P - 0.005 * P  # Growth and loss terms
    
    # Check for convergence
    if np.linalg.norm(P_new - previous_P) < tolerance:
        print(f"Converged after {iteration} iterations.")
        break
    
    previous_P = P.copy()
    P = P_new.copy()

# Plot final steady-state phytoplankton concentration
plt.figure(figsize=(6, 5))
plt.plot(P, -z, label="Steady-State Phytoplankton Concentration")
plt.xlabel("Phytoplankton Concentration (mg/m³)")
plt.ylabel("Depth (m)")
plt.title("Steady-State Phytoplankton Profile")
plt.legend()
plt.grid()
plt.show()

# Plot final light intensity profile
plt.figure(figsize=(6, 5))
plt.plot(I_values, -z, label="Steady-State Light Intensity")
plt.xlabel("Light Intensity (W/m²)")
plt.ylabel("Depth (m)")
plt.title("Steady-State Light Profile")
plt.legend()
plt.grid()
plt.show()
