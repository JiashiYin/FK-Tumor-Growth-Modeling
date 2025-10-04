import numpy as np
import os
import json

# Output directory
output_dir = '/scratch/bcsl/jyin15/Tumor/BioPhysModel-time_series/results/training'
os.makedirs(output_dir, exist_ok=True)

# Best parameters from the log (lowest loss of 0.3357)
opt_params = np.array([0.697, 0.638, 0.697, 0.995, 0.000565, -4.96, 1.0138])
min_loss = 0.3357

# Create results dictionary
results = {
    'opt_params': opt_params,
    'minLoss': min_loss,
    'time_min': 10.0,  # Approximate time
    'time_sec': 600.0  # Approximate time
}

# Save as NPY file
np.save(os.path.join(output_dir, 'best_params_results.npy'), results)

# Also save as JSON for easier inspection
with open(os.path.join(output_dir, 'best_params_results.json'), 'w') as f:
    json.dump({
        'opt_params': opt_params.tolist(),
        'minLoss': float(min_loss),
        'time_min': 10.0,
        'time_sec': 600.0
    }, f, indent=2)

print(f"Parameters saved to {output_dir}/best_params_results.npy")
print(f"Parameters also saved to {output_dir}/best_params_results.json")