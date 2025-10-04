# train_test_eval.py

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import glob
from TimeSeriesLossVisualizer import TimeSeriesLossVisualizer



def run_training(data_dir, patient_id, output_dir, train_weeks, generations=25, 
                workers=9, loss_type='combined', weighting_strategy='inverse', 
                target_time=None):
    """Run the training process using direct system call."""
    cmd = f'python TimeSeriesFitting.py --data_dir {data_dir} --patient_id {patient_id} --output_dir {output_dir} --time_points {" ".join(str(w) for w in train_weeks)} --loss_type {loss_type} --generations {generations} --workers {workers} --weighting_strategy {weighting_strategy}'
    
    # Add target_time if specified
    if target_time is not None:
        cmd += f' --target_time {target_time}'
    
    print("Running training command:")
    print(cmd)
    
    # Use direct system call which preserves stdout/stderr handling
    return_code = os.system(cmd)
    
    if return_code != 0:
        print(f"Training command failed with return code {return_code}")
        raise RuntimeError("Training command failed")
    
    

def run_forward(data_dir, patient_id, output_dir, all_weeks, param_file):
    """Run the forward model on all weeks using trained parameters."""
    forward_cmd = [
        'python', 'forward.py',
        '--data_dir', data_dir,
        '--patient_id', patient_id,
        '--output_dir', output_dir,
        '--time_points'] + [str(w) for w in all_weeks] + [
        '--param_file', param_file
    ]
    
    print("Running forward model command:")
    print(' '.join(forward_cmd))
    
    # Use Popen for real-time output streaming
    process = subprocess.Popen(
        forward_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Stream output line by line
    for line in process.stdout:
        print(line, end='', flush=True)
    
    # Wait for process to complete and check return code
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, forward_cmd)

def calculate_dice(a, b):
    """Calculate Dice coefficient between two binary masks."""
    a_bin = a > 0.5
    b_bin = b > 0.5
    
    intersection = np.sum(np.logical_and(a_bin, b_bin))
    if np.sum(a_bin) + np.sum(b_bin) == 0:
        return 0
    return 2 * intersection / (np.sum(a_bin) + np.sum(b_bin))

def evaluate_performance(data_dir, patient_id, predictions_dir, train_weeks, test_weeks):
    """Evaluate model performance on train and test weeks."""
    # Load actual and predicted tumor data
    train_dice = []
    test_dice = []
    train_loss = [] # We'll use 1-dice as the loss
    test_loss = []
    actual_train_series = []
    actual_test_series = []
    predicted_train_series = []
    predicted_test_series = []
    
    # Check for actual data (numpy files)
    for week in train_weeks:
        actual_path = os.path.join(predictions_dir, f'actual_tumor_t{week}.npy')
        pred_path = os.path.join(predictions_dir, f'predicted_tumor_t{week}.nii.gz')
        
        if os.path.exists(actual_path) and os.path.exists(pred_path):
            actual_data = np.load(actual_path)
            pred_data = nib.load(pred_path).get_fdata()
            dice = calculate_dice(actual_data, pred_data)
            train_dice.append(dice)
            train_loss.append(1 - dice)
            actual_train_series.append(actual_data)
            predicted_train_series.append(pred_data)
        else:
            print(f"Warning: Data for week {week} not found")
    
    for week in test_weeks:
        actual_path = os.path.join(predictions_dir, f'actual_tumor_t{week}.npy')
        pred_path = os.path.join(predictions_dir, f'predicted_tumor_t{week}.nii.gz')
        
        if os.path.exists(actual_path) and os.path.exists(pred_path):
            actual_data = np.load(actual_path)
            pred_data = nib.load(pred_path).get_fdata()
            dice = calculate_dice(actual_data, pred_data)
            test_dice.append(dice)
            test_loss.append(1 - dice)
            actual_test_series.append(actual_data)
            predicted_test_series.append(pred_data)
        else:
            print(f"Warning: Data for week {week} not found")
    
    # Print results
    print("\nPerformance Results:")
    print(f"Training weeks: {train_weeks}")
    for i, week in enumerate(train_weeks):
        if i < len(train_dice):
            print(f"  Week {week}: Dice = {train_dice[i]:.4f}")
    
    print(f"\nTest weeks: {test_weeks}")
    for i, week in enumerate(test_weeks):
        if i < len(test_dice):
            print(f"  Week {week}: Dice = {test_dice[i]:.4f}")
    
    mean_train_dice = np.mean(train_dice) if train_dice else 0
    mean_test_dice = np.mean(test_dice) if test_dice else 0
    
    print(f"\nMean training Dice: {mean_train_dice:.4f}")
    print(f"Mean test Dice: {mean_test_dice:.4f}")
    
    # Save results
    results = {
        'train_weeks': train_weeks,
        'test_weeks': test_weeks,
        'train_dice': train_dice,
        'test_dice': test_dice,
        'mean_train_dice': mean_train_dice,
        'mean_test_dice': mean_test_dice
    }
    
    np.save(os.path.join(predictions_dir, 'performance_results.npy'), results)
    
    # Create standard visualization
    plt.figure(figsize=(12, 8))
    
    # Collect all data points
    all_weeks = []
    all_dice = []
    
    for week, dice in zip(train_weeks, train_dice):
        all_weeks.append(week)
        all_dice.append(dice)
    
    for week, dice in zip(test_weeks, test_dice):
        all_weeks.append(week)
        all_dice.append(dice)
    
    # Sort by week
    sorted_indices = np.argsort(all_weeks)
    all_weeks = [all_weeks[i] for i in sorted_indices]
    all_dice = [all_dice[i] for i in sorted_indices]
    
    # Plot points with different colors for train and test
    for i, week in enumerate(all_weeks):
        if week in train_weeks:
            plt.plot(week, all_dice[i], 'bo', markersize=10, 
                     label='Train' if i == 0 or all_weeks[i-1] not in train_weeks else "")
        else:
            plt.plot(week, all_dice[i], 'ro', markersize=10, 
                     label='Test' if i == 0 or all_weeks[i-1] not in test_weeks else "")
    
    # Connect the points
    plt.plot(all_weeks, all_dice, 'k-', alpha=0.5)
    
    # Add mean lines
    plt.axhline(y=mean_train_dice, color='b', linestyle='--', 
                label=f'Mean Train: {mean_train_dice:.4f}')
    plt.axhline(y=mean_test_dice, color='r', linestyle='--', 
                label=f'Mean Test: {mean_test_dice:.4f}')
    
    plt.xlabel('Week', fontsize=14)
    plt.ylabel('Dice Coefficient', fontsize=14)
    plt.title('Model Performance on Training and Testing Weeks', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(predictions_dir, 'performance_comparison.png'))
    plt.close()
    
    # Create enhanced visualizations using TimeSeriesLossVisualizer
    viz_dir = os.path.join(predictions_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    visualizer = TimeSeriesLossVisualizer(save_dir=viz_dir)
    
    # Find a WM/GM file for visualization background
    wm_path = find_wm_file(data_dir, patient_id)
    if wm_path:
        wm_data = nib.load(wm_path).get_fdata()
        
        # Find a slice with tumor for visualization
        if len(actual_train_series) > 0:
            tumor_sum = np.sum(actual_train_series[0], axis=(0, 1))
            slice_index = np.argmax(tumor_sum)
        else:
            # Default to middle slice
            slice_index = wm_data.shape[2] // 2
            
        # Visualize train-test performance
        visualizer.visualize_train_test_performance(
            train_weeks, test_weeks, train_loss, test_loss
        )
        
        # If we have both training and testing data, show spatial errors
        if len(actual_train_series) > 0 and len(actual_test_series) > 0:
            all_actual_series = actual_train_series + actual_test_series
            all_predicted_series = predicted_train_series + predicted_test_series
            all_weeks = train_weeks + test_weeks
            
            # Sort by week
            sorted_indices = np.argsort(all_weeks)
            all_weeks = [all_weeks[i] for i in sorted_indices]
            all_actual_series = [all_actual_series[i] for i in sorted_indices]
            all_predicted_series = [all_predicted_series[i] for i in sorted_indices]
            
            # Visualize spatial error
            visualizer.visualize_spatial_error(
                all_predicted_series, all_actual_series, all_weeks, wm_data, slice_index
            )
            
            # Visualize volume trajectory
            visualizer.visualize_volume_trajectory(
                all_predicted_series, all_actual_series, all_weeks, voxel_volume_mm3=1.0
            )
    
    return results

def find_wm_file(data_dir, patient_id):
    """Find a white matter file in the data directory."""
    # Try different possible locations/patterns
    patterns = [
        os.path.join(data_dir, patient_id, 'T1_pve_2.nii.gz'),
        os.path.join(data_dir, patient_id, '*wm*.nii*'),
        os.path.join(data_dir, '*' + patient_id + '*wm*.nii*')
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    
    return None

def find_best_param_file(training_dir, generations):
    """Find the parameter file with the best model by examining all generation files."""
    # Look for any results files
    result_files = glob.glob(os.path.join(training_dir, '*results.npy'))
    
    if not result_files:
        return None
    
    # Find the file with the lowest loss
    best_file = None
    best_loss = float('inf')
    
    for file in result_files:
        try:
            data = np.load(file, allow_pickle=True).item()
            if 'minLoss' in data and data['minLoss'] < best_loss:
                best_loss = data['minLoss']
                best_file = file
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    if best_file:
        print(f"Found best parameters in {best_file} with loss {best_loss:.6f}")
        return best_file
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Run train/test evaluation for tumor growth modeling.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing data')
    parser.add_argument('--patient_id', type=str, required=True,
                        help='Patient ID')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--train_weeks', type=int, nargs='+', required=True,
                        help='Weeks to use for training')
    parser.add_argument('--test_weeks', type=int, nargs='+', required=True,
                        help='Weeks to use for testing')
    parser.add_argument('--generations', type=int, default=25,
                        help='Number of generations for CMA-ES')
    parser.add_argument('--workers', type=int, default=9,
                        help='Number of worker processes')
    parser.add_argument('--loss_type', type=str, default='combined',
                        choices=['dice', 'soft_dice', 'boundary', 'hausdorff', 'combined'],
                        help='Loss function to use')
    parser.add_argument('--weighting_strategy', type=str, default='inverse',
                        choices=['equal', 'exponential', 'inverse', 'log'],
                        help='Strategy for weighting time points')
    parser.add_argument('--target_time', type=int, default=None,
                        help='Target prediction time for weight calculation (default: max time point)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only run forward model and evaluation')
    parser.add_argument('--skip_forward', action='store_true',
                        help='Skip forward model and only run evaluation')
    
    args = parser.parse_args()
    
    # Create output directories
    training_dir = os.path.join(args.output_dir, 'training')
    forward_dir = os.path.join(args.output_dir, 'forward')
    
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(forward_dir, exist_ok=True)
    
    # If target_time is not specified, use the maximum test week
    if args.target_time is None and args.test_weeks:
        args.target_time = max(args.test_weeks)
        print(f"Using maximum test week ({args.target_time}) as target time for weighting.")
    
    # Step 1: Run training on training weeks
    if not args.skip_training:
        print(f"=== Step 1: Training on weeks {args.train_weeks} ===")
        run_training(
            args.data_dir, 
            args.patient_id, 
            training_dir, 
            args.train_weeks,
            args.generations,
            args.workers,
            args.loss_type,
            args.weighting_strategy,
            args.target_time
        )
    
    # Step 2: Run forward model on all weeks
    if not args.skip_forward:
        print(f"=== Step 2: Running forward model on all weeks {sorted(args.train_weeks + args.test_weeks)} ===")
        param_file = find_best_param_file(training_dir, args.generations)
        
        if not param_file:
            print(f"Error: Parameter file not found in {training_dir}.")
            print("Training may have failed or --skip_training was used without previous training.")
            return
        
        print(f"Using parameter file: {param_file}")
        
        run_forward(
            args.data_dir,
            args.patient_id,
            forward_dir,
            sorted(args.train_weeks + args.test_weeks),
            param_file
        )
    
    # Step 3: Evaluate performance
    print("=== Step 3: Evaluating performance ===")
    results = evaluate_performance(
        args.data_dir,
        args.patient_id,
        forward_dir,
        args.train_weeks,
        args.test_weeks
    )
    
    print(f"\nAll results saved to {args.output_dir}")
    
    # Show the test/train ratio as a measure of model robustness
    if results['mean_train_dice'] > 0:
        test_train_ratio = results['mean_test_dice'] / results['mean_train_dice']
        print(f"Test/Train Performance Ratio: {test_train_ratio:.2f}")
        if test_train_ratio > 0.9:
            print("✓ Model shows good generalization")
        elif test_train_ratio > 0.75:
            print("⚠ Model shows moderate generalization")
        else:
            print("⚠ Model shows signs of overfitting")

if __name__ == '__main__':
    main()