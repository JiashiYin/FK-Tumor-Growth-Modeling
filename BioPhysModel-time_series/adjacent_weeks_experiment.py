#!/usr/bin/env python3
import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def run_adjacent_weeks_experiment(data_dir, patient_id, output_base_dir, time_points, generations=13, workers=4, 
                               loss_type='soft_dice', weighting_strategy='inverse'):
    """
    Run a series of train-test experiments using pairs of adjacent time points.
    
    Args:
        data_dir: Directory containing patient data
        patient_id: Patient identifier
        output_base_dir: Base directory for saving results
        time_points: List of time points (weeks)
        generations: Number of generations for optimization
        workers: Number of parallel workers
        loss_type: Loss function type
        weighting_strategy: Strategy for weighting time points
    """
    # Create base experiment directory
    experiment_dir = os.path.join(output_base_dir, 'adjacency_experiment')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize results collection
    results = OrderedDict()
    
    # Iterate through adjacent pairs (leaving the last one for testing)
    for i in range(len(time_points) - 2):
        train_weeks = [time_points[i], time_points[i+1]]
        test_week = time_points[i+2]
        
        # Create experiment name and directory
        exp_name = f"train_{train_weeks[0]}_{train_weeks[1]}_test_{test_week}"
        exp_dir = os.path.join(experiment_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"\n\n{'='*80}")
        print(f"Running experiment: {exp_name}")
        print(f"Training on weeks {train_weeks}, testing on week {test_week}")
        print(f"{'='*80}\n")
        
        # Construct command
        cmd = [
            "python", "train_test_eval.py",
            "--data_dir", data_dir,
            "--patient_id", patient_id,
            "--output_dir", exp_dir,
            "--train_weeks", str(train_weeks[0]), str(train_weeks[1]),
            "--test_weeks", str(test_week),
            "--generations", str(generations),
            "--workers", str(workers),
            "--loss_type", loss_type,
            "--weighting_strategy", weighting_strategy
        ]
        
        # Run the command
        cmd_str = " ".join(cmd)
        print(f"Running command: {cmd_str}")
        
        try:
            subprocess.run(cmd, check=True)
            
            # Extract and store results
            train_result_path = os.path.join(exp_dir, 'training', 'best_params_results.npy')
            test_result_path = os.path.join(exp_dir, 'evaluation.json')
            
            if os.path.exists(train_result_path) and os.path.exists(test_result_path):
                # Load training results
                train_results = np.load(train_result_path, allow_pickle=True).item()
                train_loss = train_results.get('minLoss', float('inf'))
                
                # Load testing results
                with open(test_result_path, 'r') as f:
                    test_results = json.load(f)
                test_loss = test_results.get('mean_loss', float('inf'))
                test_dice = test_results.get('mean_dice', 0.0)
                
                # Store results
                results[exp_name] = {
                    'train_weeks': train_weeks,
                    'test_week': test_week,
                    'extrapolation_weeks': test_week - train_weeks[1],
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'test_dice': test_dice,
                    'loss_ratio': test_loss / train_loss if train_loss > 0 else float('inf')
                }
                
                print(f"Results for {exp_name}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Test Loss: {test_loss:.4f}")
                print(f"  Test Dice: {test_dice:.4f}")
                print(f"  Loss Ratio (Test/Train): {results[exp_name]['loss_ratio']:.4f}")
            else:
                print(f"WARNING: Results files not found for {exp_name}")
                
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Command failed with return code {e.returncode}")
            results[exp_name] = {
                'train_weeks': train_weeks,
                'test_week': test_week,
                'error': str(e)
            }
    
    # Calculate average performance
    valid_results = [r for r in results.values() if 'error' not in r]
    if valid_results:
        avg_train_loss = np.mean([r['train_loss'] for r in valid_results])
        avg_test_loss = np.mean([r['test_loss'] for r in valid_results])
        avg_test_dice = np.mean([r['test_dice'] for r in valid_results])
        avg_loss_ratio = np.mean([r['loss_ratio'] for r in valid_results])
        
        print("\n\nAverage Performance:")
        print(f"  Average Train Loss: {avg_train_loss:.4f}")
        print(f"  Average Test Loss: {avg_test_loss:.4f}")
        print(f"  Average Test Dice: {avg_test_dice:.4f}")
        print(f"  Average Loss Ratio (Test/Train): {avg_loss_ratio:.4f}")
    
    # Create summary visualizations
    create_summary_visualizations(results, experiment_dir)
    
    # Save results to JSON
    results_path = os.path.join(experiment_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        # Convert OrderedDict to regular dict for JSON serialization
        json.dump({k: v for k, v in results.items()}, f, indent=2)
    
    print(f"\nExperiment results saved to {results_path}")
    
    # Run parameter recovery on the experiment directory as a fallback
    run_parameter_recovery(experiment_dir)
    
    return results

def create_summary_visualizations(results, output_dir):
    """Create summary visualizations of experiment results."""
    # Filter out experiments with errors
    valid_results = OrderedDict((k, v) for k, v in results.items() if 'error' not in v)
    if not valid_results:
        print("No valid results to visualize")
        return
    
    # Extract data for plotting
    experiment_names = list(valid_results.keys())
    train_losses = [r['train_loss'] for r in valid_results.values()]
    test_losses = [r['test_loss'] for r in valid_results.values()]
    extrapolation_weeks = [r['extrapolation_weeks'] for r in valid_results.values()]
    test_dice = [r['test_dice'] for r in valid_results.values()]
    loss_ratios = [r['loss_ratio'] for r in valid_results.values()]
    
    # Create a summary figure with multiple subplots
    plt.figure(figsize=(14, 10))
    
    # 1. Train vs Test Loss
    plt.subplot(2, 2, 1)
    x = np.arange(len(experiment_names))
    width = 0.35
    plt.bar(x - width/2, train_losses, width, label='Train Loss')
    plt.bar(x + width/2, test_losses, width, label='Test Loss')
    plt.xlabel('Experiment')
    plt.ylabel('Loss')
    plt.title('Training vs Testing Loss')
    plt.xticks(x, [f"{res['train_weeks']}→{res['test_week']}" for res in valid_results.values()], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Test Dice Score
    plt.subplot(2, 2, 2)
    plt.bar(x, test_dice, color='green')
    plt.xlabel('Experiment')
    plt.ylabel('Dice Score')
    plt.title('Test Dice Score (higher is better)')
    plt.xticks(x, [f"{res['train_weeks']}→{res['test_week']}" for res in valid_results.values()], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. Loss Ratio (Test/Train)
    plt.subplot(2, 2, 3)
    plt.bar(x, loss_ratios, color='red')
    plt.xlabel('Experiment')
    plt.ylabel('Loss Ratio (Test/Train)')
    plt.title('Loss Ratio (lower is better)')
    plt.xticks(x, [f"{res['train_weeks']}→{res['test_week']}" for res in valid_results.values()], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. Scatter plot of extrapolation distance vs test loss
    plt.subplot(2, 2, 4)
    plt.scatter(extrapolation_weeks, test_losses)
    plt.xlabel('Extrapolation Distance (weeks)')
    plt.ylabel('Test Loss')
    plt.title('Extrapolation Distance vs Test Loss')
    
    # Show average values for test metrics
    avg_test_loss = np.mean(test_losses)
    avg_test_dice = np.mean(test_dice)
    avg_loss_ratio = np.mean(loss_ratios)
    
    plt.figtext(0.5, 0.01, 
                f"Average Test Loss: {avg_test_loss:.4f} | "
                f"Average Test Dice: {avg_test_dice:.4f} | "
                f"Average Loss Ratio: {avg_loss_ratio:.4f}",
                ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_results.png'), dpi=200)
    plt.close()
    
    print(f"Created summary visualization at {os.path.join(output_dir, 'summary_results.png')}")

def run_parameter_recovery(experiment_dir):
    """Run parameter recovery on experiment directory."""
    recovery_script = "recover_parameters.py"
    try:
        print(f"Running parameter recovery on {experiment_dir}")
        subprocess.run(["python", recovery_script, experiment_dir], check=True)
        print("Parameter recovery completed successfully")
    except Exception as e:
        print(f"Parameter recovery failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run adjacent weeks train-test experiments')
    parser.add_argument('--data_dir', required=True, help='Directory containing patient data')
    parser.add_argument('--patient_id', required=True, help='Patient ID')
    parser.add_argument('--output_dir', required=True, help='Base output directory')
    parser.add_argument('--weeks', nargs='+', type=int, required=True, help='List of time points (weeks)')
    parser.add_argument('--generations', type=int, default=13, help='Number of generations for optimization')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--loss_type', default='soft_dice', help='Loss function type')
    parser.add_argument('--weighting_strategy', default='inverse', help='Weighting strategy')
    
    args = parser.parse_args()
    
    # Run the experiment
    run_adjacent_weeks_experiment(
        args.data_dir,
        args.patient_id,
        args.output_dir,
        args.weeks,
        args.generations,
        args.workers,
        args.loss_type,
        args.weighting_strategy
    )