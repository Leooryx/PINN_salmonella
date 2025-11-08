import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
filename = "synthetic_training_data"
output_dir = "data/synthetic_data"
os.makedirs(output_dir, exist_ok=True)

## Definition of the ODE system developed during the intership of M. Denoual
def ode_system(t, y, params):
    '''Defines the ODE system for the biological model.
    Args:
        t (float): Time variable.
        y (list): List of state variables [S, A, B, i, m].
        params (dict): Dictionary of model parameters.
    Returns:
        list: Derivatives [dS, dA, dB, di, dm].
    '''
    S, A, B, i, m = y
    mu1, K_S, C_SA, C_SB = params["mu1"], params["K_S"], params["C_SA"], params["C_SB"]
    mu2, K_A, C_AS, C_AB = params["mu2"], params["K_A"], params["C_AS"], params["C_AB"]
    mu3, K_B, C_BS, C_BA = params["mu3"], params["K_B"], params["C_BS"], params["C_BA"]
    beta_S, beta_A, beta_B = params["beta_S"], params["beta_A"], params["beta_B"]
    gamma1, gamma2, gamma3 = params["gamma1"], params["gamma2"], params["gamma3"]
    alpha_S, alpha_A = params["alpha_S"], params["alpha_A"]
    K_I, n_I = params["K_I"], params["n_I"]
    delta_i, C_iB, C_im = params["delta_i"], params["C_iB"], params["C_im"]
    rho = params["rho"]

    fS = i * (beta_S - m)
    fA = i * (beta_A - m)
    fB = i * (beta_B - m)

    dS = S * (mu1 * (1 - (S + C_SA * A + C_SB * B) / K_S) + gamma1 * fS)
    dA = A * (mu2 * (1 - (A + C_AS * S + C_AB * B) / K_A) - gamma2 * fA)
    dB = B * (mu3 * (1 - (B + C_BS * S + C_BA * A) / K_B) - gamma3 * fB)

    H_S = S**n_I / (K_I**n_I + S**n_I)
    H_A = A**n_I / (K_I**n_I + A**n_I)
    di = alpha_S * H_S + alpha_A * H_A - (delta_i + C_iB * B + C_im * m) * i

    dm = rho * i * (1 - m)

    return [dS, dA, dB, di, dm]

# Parameters
params = {
    'mu1': 2.0, 'K_S': 300.0, 'C_SA': 0.5,  'C_SB': 0.9,
    'mu2': 1.5, 'K_A': 400.0, 'C_AS': 0.6,  'C_AB': 0.2,
    'mu3': 1.5, 'K_B': 300.0, 'C_BS': 1.1,  'C_BA': 0.2,
    'beta_S': 0.7, 'beta_A': 1.2, 'beta_B': 1.3,
    'gamma1': 0.3, 'gamma2': 0.1, 'gamma3': 0.1,
    'alpha_S': 6.0, 'alpha_A': 3.0, 'K_I': 1.0, 'n_I': 20.0,
    'delta_i': 0.1, 'C_iB': 0.001, 'C_im': 0.8,
    'rho': 0.025
}

# Initial conditions
y0 = [100.0, 100.0, 80.0, 0.8, 0.2]

# Time points for evaluation
t_eval = np.unique(np.concatenate([
    np.linspace(0, 27, 200),
    np.array([4, 6, 11, 14, 20, 27])
]))

# Simulation
sol = solve_ivp(lambda t, y: ode_system(t, y, params), [0, 27], y0, t_eval=t_eval, method="LSODA")

color_list = ['blue', 'orange', 'green', 'red', 'purple']

# Generating synthetic noisy data
noise_level = 0.05      # 5% noise
synthetic_data = sol.y + noise_level * np.random.randn(*sol.y.shape) * sol.y

# Desired time points for data extraction
t_targets = np.array([0,4, 6, 11, 14, 20, 27])
# Find the corresponding indices in sol.t
indices = [np.argmin(np.abs(sol.t - tt)) for tt in t_targets]
# Corresponding times
t_data = sol.t[indices]
# Corresponding noisy values
y_data = synthetic_data[:, indices].T

# Creating a DataFrame for saving
df = pd.DataFrame(
    y_data,
    columns=["S", "A", "B", "i", "m"]
)
df.insert(0, "time", t_data)  # adds the time column at the beginning

# Saving as CSV
df.to_csv(os.path.join(output_dir, filename+".csv"), index=False)
print(f"synthetic_training_data.csv saved in {output_dir}")
print(df)

# Visualization
fig, ax = plt.subplot_mosaic([
    [0, 1],
    [0, 2]
], figsize=(13.5, 7))
ax[0].plot(sol.t, sol.y[0], label="S", color=color_list[0])
ax[0].plot(sol.t, sol.y[1], label="A", color=color_list[1])
ax[0].plot(sol.t, sol.y[2], label="B", color=color_list[2])
ax[1].plot(sol.t, sol.y[3], label="i", color=color_list[3])
ax[2].plot(sol.t, sol.y[4], label="m", color=color_list[4])
for a in ax.values():
    a.set_xlabel("Time (s)")
    a.legend()
ax[0].set_ylabel("Population")
ax[1].set_ylabel("Inflammation intensity")
ax[2].set_ylabel("Host immune system maturity")

ax[0].plot(t_data, y_data[:,0], 'x', color=color_list[0])
ax[0].plot(t_data, y_data[:,1], 'x', color=color_list[1])
ax[0].plot(t_data, y_data[:,2], 'x', color=color_list[2])
ax[1].plot(t_data, y_data[:,3], 'x', color=color_list[3])
ax[2].plot(t_data, y_data[:,4], 'x', color=color_list[4])
#
plt.tight_layout()
plt.show()