import numpy as np

# Load the results file
results = np.load('gen_12_results.npy', allow_pickle=True).item()

# Extract optimized parameters
opt_params = results['opt_params']

# Map the optimized parameters
params = {
    'Dw': opt_params[4],               # Diffusion coefficient for white matter
    'rho': opt_params[3],              # Tumor proliferation rate
    'NxT1_pct': opt_params[0],         # Tumor position X (percentage)
    'NyT1_pct': opt_params[1],         # Tumor position Y (percentage)
    'NzT1_pct': opt_params[2]          # Tumor position Z (percentage)
}

# Print the extracted parameters
print("Optimized Parameters for Forward Simulation:")
print(params)
