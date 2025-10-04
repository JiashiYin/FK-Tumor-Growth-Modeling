# evaluate_predictions.py
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
import argparse
from scipy import ndimage

def calculate_dice(a, b):
    """Calculate Dice coefficient between two binary masks."""
    a_bin = a > 0.5
    b_bin = b > 0.5
    
    intersection = np.sum(np.logical_and(a_bin, b_bin))
    if np.sum(a_bin) + np.sum(b_bin) == 0:
        return 0
    return 2 * intersection / (np.sum(a_bin) + np.sum(b_bin))

def calculate_hausdorff(a, b, percentile=95):
    """Calculate modified Hausdorff distance between two binary masks."""
    a_bin = a > 0.5
    b_bin = b > 0.5
    
    # If either mask is empty, return max distance
    if np.sum(a_bin) == 0 or np.sum(b_bin) == 0:
        return float('inf')
    
    # Calculate distance transforms
    dist_a = ndimage.distance_transform_edt(~a_bin)
    dist_b = ndimage.distance_transform_edt(~b_bin)
    
    # Get distances from each point in A to the closest point in B
    distances_a_to_b = dist_b[a_bin]
    distances_b_to_a = dist_a[b_bin]
    
    # Calculate percentile distances
    h1 = np.percentile(distances_a_to_b, percentile)
    h2 = np.percentile(distances_b_to_a, percentile)
    
    # Take the maximum of the two distances
    return max(h1, h2)

def calculate_volume_error(a, b, voxel_size_mm3=1.0):
    """Calculate volume error as a percentage of actual volume."""
    a_bin = a > 0.5
    b_bin = b > 0.5
    
    vol_a = np.sum(a_bin) * voxel_size_mm3
    vol_b = np.sum(b_bin) * voxel_size_mm3
    
    if vol_b == 0:
        return float('inf') if vol_a > 0 else 0
    
    return abs(vol_a - vol_b) / vol_b * 100  # Percentage error

def load_data(data_dir, week):
    """Load predicted and actual tumor data for a specific week."""
    actual_path = os.path.join(data_dir, f'actual_tumor_t{week}.npy')
    pred_nifti_path = os.path.join(data_dir, f'predicted_tumor_t{week}.nii.gz')
    pred_npy_path = os.path.join(data_dir, f'predicted_tumor_t{week}.npy')
    
    # Try to load actual tumor data
    if os.path.exists(actual_path):
        actual_data = np.load(actual_path)
    else:
        print(f"Warning: Actual tumor data for week {week} not found")
        actual_data = None
    
    # Try to load predicted tumor data (try NIFTI first, then NPY)
    if os.path.exists(pred_nifti_path):
        pred_data = nib.load(pred_nifti_path).get_fdata()
    elif os.path.exists(pred_npy_path):
        pred_data = np.load(pred_npy_path)
    else:
        print(f"Warning: Predicted tumor data for week {week} not found")
        pred_data = None
    
    return actual_data, pred_data

def evaluate_and_visualize(data_dir, output_dir, train_weeks, test_weeks, wm_data_path=None):
    """Evaluate model performance and generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data for all weeks
    week_data = {}
    for week in train_weeks + test_weeks:
        actual, pred = load_data(data_dir, week)
        if actual is not None and pred is not None:
            week_data[week] = {'actual': actual, 'pred': pred}
    
    if not week_data:
        print("Error: No valid data found for any week")
        return
    
    # Calculate metrics for each week
    metrics = {}
    for week, data in week_data.items():
        metrics[week] = {
            'dice': calculate_dice(data['actual'], data['pred']),
            'hausdorff': calculate_hausdorff(data['actual'], data['pred']),
            'volume_error': calculate_volume_error(data['actual'], data['pred'])
        }
    
    # Separate metrics for train and test weeks
    train_metrics = {week: metrics[week] for week in train_weeks if week in metrics}
    test_metrics = {week: metrics[week] for week in test_weeks if week in metrics}
    
    # Calculate average metrics
    avg_train_dice = np.mean([m['dice'] for m in train_metrics.values()]) if train_metrics else 0
    avg_test_dice = np.mean([m['dice'] for m in test_metrics.values()]) if test_metrics else 0
    
    avg_train_hausdorff = np.mean([m['hausdorff'] for m in train_metrics.values()]) if train_metrics else 0
    avg_test_hausdorff = np.mean([m['hausdorff'] for m in test_metrics.values()]) if test_metrics else 0
    
    avg_train_vol_err = np.mean([m['volume_error'] for m in train_metrics.values()]) if train_metrics else 0
    avg_test_vol_err = np.mean([m['volume_error'] for m in test_metrics.values()]) if test_metrics else 0
    
    # Print results
    print("\n=== Performance Results ===")
    print(f"Training weeks: {sorted(train_metrics.keys())}")
    for week in sorted(train_metrics.keys()):
        print(f"  Week {week}: Dice = {train_metrics[week]['dice']:.4f}, "
              f"Hausdorff = {train_metrics[week]['hausdorff']:.2f}, "
              f"Vol Err = {train_metrics[week]['volume_error']:.2f}%")
    
    print(f"\nTest weeks: {sorted(test_metrics.keys())}")
    for week in sorted(test_metrics.keys()):
        print(f"  Week {week}: Dice = {test_metrics[week]['dice']:.4f}, "
              f"Hausdorff = {test_metrics[week]['hausdorff']:.2f}, "
              f"Vol Err = {test_metrics[week]['volume_error']:.2f}%")
    
    print(f"\nAverage Training Metrics: Dice = {avg_train_dice:.4f}, "
          f"Hausdorff = {avg_train_hausdorff:.2f}, Vol Err = {avg_train_vol_err:.2f}%")
    print(f"Average Test Metrics: Dice = {avg_test_dice:.4f}, "
          f"Hausdorff = {avg_test_hausdorff:.2f}, Vol Err = {avg_test_vol_err:.2f}%")
    
    # Save metrics to file
    results = {
        'train_weeks': train_weeks,
        'test_weeks': test_weeks,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'avg_train_dice': avg_train_dice,
        'avg_test_dice': avg_test_dice,
        'avg_train_hausdorff': avg_train_hausdorff,
        'avg_test_hausdorff': avg_test_hausdorff,
        'avg_train_vol_err': avg_train_vol_err,
        'avg_test_vol_err': avg_test_vol_err
    }
    
    np.save(os.path.join(output_dir, 'evaluation_results.npy'), results)
    
    # Create Dice score plot
    all_weeks = sorted(metrics.keys())
    dice_values = [metrics[week]['dice'] for week in all_weeks]
    
    plt.figure(figsize=(12, 6))
    for i, week in enumerate(all_weeks):
        if week in train_weeks:
            plt.plot(week, dice_values[i], 'bo', markersize=10, label='Train' if i == 0 else "")
        else:
            plt.plot(week, dice_values[i], 'ro', markersize=10, label='Test' if week == test_weeks[0] else "")
    
    # Connect the points
    plt.plot(all_weeks, dice_values, 'k-', alpha=0.5)
    
    # Add horizontal lines for average dice
    if train_metrics:
        plt.axhline(y=avg_train_dice, color='b', linestyle='--', 
                  label=f'Mean Train: {avg_train_dice:.4f}')
    if test_metrics:
        plt.axhline(y=avg_test_dice, color='r', linestyle='--', 
                  label=f'Mean Test: {avg_test_dice:.4f}')
    
    plt.xlabel('Week', fontsize=14)
    plt.ylabel('Dice Coefficient', fontsize=14)
    plt.title('Model Performance on Training and Testing Weeks', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(all_weeks)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_performance.png'), dpi=300)
    plt.close()
    
    # Create volume comparison plot
    if all_weeks:
        actual_volumes = [np.sum(week_data[week]['actual'] > 0.5) for week in all_weeks]
        pred_volumes = [np.sum(week_data[week]['pred'] > 0.5) for week in all_weeks]
        
        plt.figure(figsize=(12, 6))
        plt.plot(all_weeks, actual_volumes, 'bs-', label='Actual Volume', linewidth=2)
        plt.plot(all_weeks, pred_volumes, 'ro-', label='Predicted Volume', linewidth=2)
        
        plt.xlabel('Week', fontsize=14)
        plt.ylabel('Tumor Volume (voxels)', fontsize=14)
        plt.title('Actual vs Predicted Tumor Volume Over Time', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xticks(all_weeks)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'volume_comparison.png'), dpi=300)
        plt.close()
    
    # Create spatial visualization of predictions vs actual
    # This requires loading WM data for visualization background if provided
    if wm_data_path and os.path.exists(wm_data_path):
        try:
            wm_data = nib.load(wm_data_path).get_fdata()
            
            # Find a good slice to visualize (with tumor)
            sample_tumor = list(week_data.values())[0]['actual']
            tumor_sum = np.sum(sample_tumor, axis=(0, 1))
            slice_idx = np.argmax(tumor_sum)
            
            # Create visualization grid
            n_weeks = len(all_weeks)
            fig, axes = plt.subplots(2, n_weeks, figsize=(5*n_weeks, 8))
            
            for i, week in enumerate(all_weeks):
                # Show actual tumor
                axes[0, i].imshow(wm_data[:, :, slice_idx], cmap='gray')
                axes[0, i].imshow(week_data[week]['actual'][:, :, slice_idx], cmap='hot', alpha=0.7)
                axes[0, i].set_title(f"Actual Week {week}", fontsize=12)
                axes[0, i].axis('off')
                
                # Show predicted tumor
                axes[1, i].imshow(wm_data[:, :, slice_idx], cmap='gray')
                axes[1, i].imshow(week_data[week]['pred'][:, :, slice_idx], cmap='hot', alpha=0.7)
                axes[1, i].set_title(f"Predicted Week {week}\nDice: {metrics[week]['dice']:.4f}", fontsize=12)
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'spatial_comparison.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create spatial visualization: {str(e)}")
    
    print(f"\nResults saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate tumor growth predictions.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing prediction and actual tumor data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--train_weeks', type=int, nargs='+', required=True,
                        help='Weeks used for training')
    parser.add_argument('--test_weeks', type=int, nargs='+', required=True,
                        help='Weeks used for testing')
    parser.add_argument('--wm_data', type=str, default=None,
                        help='Path to white matter data for visualization (optional)')
    
    args = parser.parse_args()
    
    evaluate_and_visualize(
        args.data_dir,
        args.output_dir,
        args.train_weeks,
        args.test_weeks,
        args.wm_data
    )

if __name__ == '__main__':
    main()
