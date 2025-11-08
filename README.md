# Salmonella dynamic modeling using Physics-Informed Neural Networks (PINNs) - codes folder
A research project implementing Physics-Informed Neural Networks for parameter inference in biological systems, specifically focused on Salmonella dynamics in host immune system interactions.

## Project Overview
This project develops and applies PINNs to perform **parameter inference** from observational data, given a known ODE model that describes the complex dynamics between:
- **S**: Salmonella enterica population
- **A**: Facultative anaerobic bacteria population  
- **B**: Strict anaerobic bacteria population
- **i**: Inflammation intensity
- **m**: Barrier effect maturation (immunity, mucus, microbiota)

The biological model was developed during the internship of M. Denoual (2025) and captures the ecological interactions between bacterial populations and their modulation by host immune responses. 


## Project Structure
```
codes/
├── create_synthetic_data.py    # Generate synthetic training data from ODE solutions
├── forward_pinn.py             # PINN implementation for testing architecture on the forward problem
├── PINN_example.py             # Simple PINN example for learning/testing
├── requirements.txt            # Python dependencies
└── README.md
```

## Mathematical Model
The ODE system models the dynamics:

$$\begin{align}
\frac{dS}{dt} &= S\left(\mu_1\left(1 - \frac{S + C_{SA} \cdot A + C_{SB} \cdot B}{K_S}\right) + \gamma_1 \cdot f_S\right) \\
\frac{dA}{dt} &= A\left(\mu_2\left(1 - \frac{A + C_{AS} \cdot S + C_{AB} \cdot B}{K_A}\right) - \gamma_2 \cdot f_A\right) \\
\frac{dB}{dt} &= B\left(\mu_3\left(1 - \frac{B + C_{BS} \cdot S + C_{BA} \cdot A}{K_B}\right) - \gamma_3 \cdot f_B\right) \\
\frac{di}{dt} &= \alpha_S \cdot H_S + \alpha_A \cdot H_A - (\delta_i + C_{iB} \cdot B + C_{im} \cdot m) \cdot i \\
\frac{dm}{dt} &= \rho \cdot i \cdot (1 - m)
\end{align}$$

Where:
- $f_X = i(\beta_X - m)$ represents immune modulation
- $H_X = \frac{X^{n_I}}{K_I^{n_I} + X^{n_I}}$ are Hill functions
- Parameters control growth rates, carrying capacities, and interaction strengths

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Key dependencies:
- `jinns`: PINN framework
- `jax`: Automatic differentiation
- `equinox`: Neural network library
- `optax`: Optimization
- `matplotlib`: Visualization
- `pandas`: Data handling
- `scipy`: Numerical integration

## Usage

### 1. Generate Synthetic Data
```bash
python create_synthetic_data.py
```
This creates synthetic "experimental" data with noisy observations at key time points, simulating what would be available for parameter inference in a real scenario.

### 2. Test PINN Architecture (Forward Problem)
```bash
python forward_pinn.py
```
This will:
- Initialize the PINN architecture with known parameters
- Train the model to solve the forward problem using physics constraints
- Test the PINN framework and architecture
- Generate predictions for the known ODE system
- Display training loss curves and model performance

### 3. Simple PINN Example
```bash
python PINN_example.py
```
A basic PINN implementation for learning the framework.

