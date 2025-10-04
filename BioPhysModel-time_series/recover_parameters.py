#!/usr/bin/env python3
import os
import re
import sys
import numpy as np
import glob

def extract_params_from_log(log_file):
    """Extract the best parameters from a log file."""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return None, None
    
    best_params = None
    best_loss = float('inf')
    
    with open(log_file, 'r') as f:
        content = f.read()
        
    # Find all parameter lines: Parameters: [x.xxx, y.yyy, ...], Loss: 0.xxxx
    pattern = r"Parameters: \[([\d\.\-e, ]+)\], Loss: ([\d\.e\-]+)"
    matches = re.findall(pattern, content)
    
    # Extract the parameters with the lowest loss
    for params_str, loss_str in matches:
        try:
            params = [float(x.strip()) for x in params_str.split(',')]
            loss = float(loss_str)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params
        except Exception as e:
            print(f"Error parsing parameter line: {e}")
            continue
    
    if best_params:
        print(f"Found best parameters with loss {best_loss}: {best_params}")
        return best_params, best_loss
    else:
        print("No valid parameters found in log")
        return None, None

def create_params_file(params, loss, output_file):
    """Create a parameter file with the given parameters and loss."""
    if params is None or loss is None:
        return False
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create results dictionary
    results = {
        'opt_params': np.array(params),
        'minLoss': loss,
        'time_min': 10.0,
        'time_sec': 600.0
    }
    
    # Save to NPY file
    np.save(output_file, results)
    print(f"Parameter file created: {output_file}")
    return True

def process_experiment_dir(experiment_dir):
    """Process a specific experiment directory."""
    # Find log files
    log_patterns = [
        os.path.join(experiment_dir, "*.log"),
        os.path.join(experiment_dir, "training", "*.log"),
        os.path.join(experiment_dir, "logs", "*.log")
    ]
    
    log_files = []
    for pattern in log_patterns:
        log_files.extend(glob.glob(pattern))
    
    # Try standard output if no log files
    if not log_files:
        stdout_file = os.path.join(experiment_dir, "stdout.txt")
        if os.path.exists(stdout_file):
            log_files.append(stdout_file)
    
    # Process each log file
    for log_file in log_files:
        print(f"Processing log file: {log_file}")
        params, loss = extract_params_from_log(log_file)
        
        if params is not None:
            # Create parameter files in expected locations
            paths = [
                os.path.join(experiment_dir, 'training', 'best_params_results.npy'),
                os.path.join(experiment_dir, 'best_params_results.npy')
            ]
            
            for path in paths:
                if not os.path.exists(path):
                    create_params_file(params, loss, path)

def main():
    if len(sys.argv) < 2:
        print("Usage: python recover_parameters.py <experiment_dir>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if os.path.isdir(experiment_dir):
        # Check if it's the base adjacency experiment dir
        adjacency_dir = os.path.join(experiment_dir, 'adjacency_experiment')
        if os.path.isdir(adjacency_dir):
            print(f"Processing adjacency experiment directory: {adjacency_dir}")
            # Process each experiment subdirectory
            for entry in os.listdir(adjacency_dir):
                subdir = os.path.join(adjacency_dir, entry)
                if os.path.isdir(subdir) and entry.startswith('train_'):
                    print(f"\nProcessing experiment: {entry}")
                    process_experiment_dir(subdir)
        else:
            # Process a single experiment directory
            process_experiment_dir(experiment_dir)
    else:
        print(f"Directory not found: {experiment_dir}")
        sys.exit(1)

if __name__ == "__main__":
    main()