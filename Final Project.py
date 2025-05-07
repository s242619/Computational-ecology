import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Common Parameters
e_a = 0.69
aI = -0.28
aM = -0.28
A = -14.05
B = 9.38
C = 0.46
D = 0.17
T_low = 2
T_high = 22
initial_weight = 50  # g
w_max = 3000 #g 
ref_T = 15  # Reference temperature

# Functions for temperature-dependent coefficients
def C_I(T):
    return A + B * np.log(T) if T_low < T < T_high else 0

def C_M(T):
    return C * np.exp(D * T)

# Scale CI and CM at reference T and weight
CI_ref = C_I(ref_T)
CM_ref = C_M(ref_T)
I_target = 0.05
M_target = 0.02
I_model = CI_ref * initial_weight**aI
M_model = CM_ref * initial_weight**aM
CI_ref *= I_target / I_model
CM_ref *= M_target / M_model

# ODE: 
def fish_growth(t, w):
    CI = C_I(ref_T) * (CI_ref / C_I(ref_T))
    CM = C_M(ref_T) * (CM_ref / C_M(ref_T))
    I = CI * w[0]**aI
    M = CM * w[0]**aM
    growth = (e_a * I - M) * w[0]
    logistic_term = 1 - (w[0] / w_max)
    return [growth * logistic_term]

# Generalized growth model with variable T 
def growth_model(t, w, T):
    if w[0] <= 0:
        return [0]
    CI = C_I(T) * (CI_ref / C_I(ref_T))
    CM = C_M(T) * (CM_ref / C_M(ref_T))
    I = CI* w[0]**aI 
    M = CM * w[0]**aM
    growth = (e_a * I - M) * w[0] * (1 - w[0]/w_max)
    return [growth]
def growth_model_I(t, w, T, I_val):
    if w[0] <= 0:
        return [0]
    CM = C_M(T) * (CM_ref / C_M(ref_T))
    M = CM * w[0]**aM
    I = I_val  # fixed feeding rate in g/g/day
    growth = (e_a * I - M) * w[0] * (1 - w[0]/w_max)
    return [growth]


# Time settings
t_span = (0, 1500)
t_eval = np.linspace(*t_span, 1500)

# === Plot 1: 
sol = solve_ivp(fish_growth, t_span, [initial_weight], t_eval=t_eval)

plt.figure(figsize=(8, 6))
plt.plot(sol.t, sol.y[0], label='Fish Weight Over Time (T = 15°C)', color='blue', lw=2)
plt.xlabel('Time (days)')
plt.ylabel('Fish Weight (g)')
plt.title('Fish Growth at T = 15°C')
plt.grid(True)
plt.legend()
plt.ylim(0, 4000)
plt.tight_layout()
plt.show()

# === Plot 2: Temperature effects ===
temps = [4, 10, 15, 20]
colors_temp = ['blue', 'green', 'red', 'purple']
labels_temp = ['Temp (4°C)', 'Temp (10°C)', 'Temp (15°C)', 'Temp (20°C)']


plt.figure(figsize=(10, 6))
for T, color, label in zip(temps, colors_temp, labels_temp):
    sol = solve_ivp(growth_model, t_span, [initial_weight], t_eval=t_eval, args=(T,))
    linestyle = '--' if T == 10 else '-' if T == 15 else ':'
    plt.plot(sol.t, sol.y[0], color=color, label=label, lw=2, linestyle=linestyle)
plt.xlabel('Time (days)')
plt.ylabel('Fish Weight (g)')
plt.title('Effect of Temperature on Fish Growth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
I_values = [0.01, 0.03, 0.05]  # actual feeding rates
colors_feed = ['orange', 'green', 'purple']
labels_feed = ['Feeding rate = 0.01 g/g/day', 'Feeding rate = 0.03 g/g/day', 'Feeding rate = 0.05 g/g/day']
T_fixed = 15

plt.figure(figsize=(10, 6))
for I_val, color, label in zip(I_values, colors_feed, labels_feed):
    sol = solve_ivp(growth_model_I, t_span, [initial_weight], t_eval=t_eval, args=(T_fixed, I_val))
    linestyle = '--' if I_val == 0.01 else '-' if I_val == 0.03 else ':'
    plt.plot(sol.t, sol.y[0], label=label, color=color, lw=2, linestyle=linestyle)

plt.xlabel('Time (days)')
plt.ylabel('Fish Weight (g)')
plt.title('Effect of Feeding Rate on Fish Growth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
