# TimeSeriesLossVisualizer.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D

class TimeSeriesLossVisualizer:
    def __init__(self, save_dir="visualization_results"):
        """Initialize the visualizer with a save directory."""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_all(self, result_dict, predicted_series, actual_series, time_points, 
                      wm_data, slice_index, voxel_volume_mm3=1.0):
        """Generate all visualizations for the time series fitting."""
        # Extract individual loss values if available
        individual_losses = []
        for gen_losses in result_dict.get("lossDir", []):
            for sample_loss in gen_losses:
                if "individual_losses" in sample_loss and sample_loss.get("total_loss") == result_dict.get("minLoss"):
                    individual_losses = sample_loss["individual_losses"]
                    break
            if individual_losses:
                break
        
        if not individual_losses and len(predicted_series) == len(actual_series):
            # Recalculate individual losses using Dice coefficient
            individual_losses = [1 - self._calculate_dice(pred, actual) 
                               for pred, actual in zip(predicted_series, actual_series)]
        
        # Generate all visualizations
        self.visualize_timepoint_losses(individual_losses, time_points)
        self.plot_optimization_convergence(result_dict)
        self.visualize_parameter_evolution(result_dict)
        self.visualize_spatial_error(predicted_series, actual_series, time_points, 
                                    wm_data, slice_index)
        self.visualize_volume_trajectory(predicted_series, actual_series, time_points, 
                                        voxel_volume_mm3)
        
        # Create additional violin plot of parameter distribution
        self.visualize_parameter_distribution(result_dict)
        
        print(f"All visualizations saved to {self.save_dir}")
    
    def _calculate_dice(self, a, b):
        """Calculate Dice coefficient between two tumor masks."""
        a_bin = a > 0.5
        b_bin = b > 0.5
        
        intersection = np.sum(np.logical_and(a_bin, b_bin))
        if np.sum(a_bin) + np.sum(b_bin) == 0:
            return 0
        return 2 * intersection / (np.sum(a_bin) + np.sum(b_bin))
    
    def visualize_timepoint_losses(self, individual_losses, time_points):
        """Visualize loss for each time point."""
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, individual_losses, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Time (weeks)', fontsize=14)
        plt.ylabel('Loss Value', fontsize=14)
        plt.title('Model Loss Across Time Points', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add value labels above each point
        for i, loss in enumerate(individual_losses):
            plt.text(time_points[i], loss + 0.02, f"{loss:.3f}", 
                    ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'timepoint_losses.png'), dpi=300)
        plt.close()
    
    def plot_optimization_convergence(self, result_dict):
        """Plot the convergence of the optimization process."""
        if not result_dict.get("y0s"):
            print("Warning: No optimization data available for convergence plot")
            return
            
        generations = len(result_dict["y0s"])
        best_fitness = [min(y) for y in result_dict["y0s"]]
        mean_fitness = [np.mean(y) for y in result_dict["y0s"]]
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, generations+1), best_fitness, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(range(1, generations+1), mean_fitness, 'r--', label='Mean Fitness', linewidth=2)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Loss Value (lower is better)', fontsize=14)
        plt.title('CMA-ES Optimization Convergence', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Annotate final best value
        plt.annotate(f'Final best: {best_fitness[-1]:.4f}', 
                    xy=(generations, best_fitness[-1]),
                    xytext=(generations-3, best_fitness[-1]+0.05),
                    fontsize=12, 
                    arrowprops=dict(arrowstyle='->'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'convergence.png'), dpi=300)
        plt.close()
    
    def visualize_parameter_evolution(self, result_dict):
        """Visualize how parameters evolved during optimization."""
        if not result_dict.get("xmeans"):
            print("Warning: No parameter evolution data available")
            return
            
        param_names = ['NxT1_pct', 'NyT1_pct', 'NzT1_pct', 'Dw', 'rho']
        param_indices = range(len(param_names))
        
        plt.figure(figsize=(15, 10))
        for idx, name in zip(param_indices, param_names):
            plt.subplot(len(param_names), 1, idx+1)
            
            # Extract parameter values across generations
            param_values = [xmean[idx] for xmean in result_dict["xmeans"]]
            
            plt.plot(range(1, len(param_values)+1), param_values, 'g-', linewidth=2)
            plt.ylabel(name, fontsize=12)
            plt.grid(True, alpha=0.3)
            
            if idx == 0:
                plt.title('Parameter Evolution During Optimization', fontsize=16)
            if idx == len(param_names)-1:
                plt.xlabel('Generation', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'parameter_evolution.png'), dpi=300)
        plt.close()
    
    def visualize_parameter_distribution(self, result_dict):
        """Visualize distribution of parameter samples across generations."""
        if not result_dict.get("xs0s"):
            print("Warning: No parameter samples available for distribution plot")
            return
            
        param_names = ['NxT1_pct', 'NyT1_pct', 'NzT1_pct', 'Dw', 'rho']
        
        # Collect all parameter samples
        all_samples = []
        for gen in result_dict["xs0s"]:
            all_samples.extend(gen)
        
        if not all_samples:
            print("Warning: No parameter samples available")
            return
            
        all_samples = np.array(all_samples)
        
        # Create violin plots for each parameter
        plt.figure(figsize=(12, 8))
        violin_parts = plt.violinplot([all_samples[:, i] for i in range(len(param_names))], 
                                     showmeans=True, showmedians=True)
        
        # Customize violin plot
        for pc in violin_parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        violin_parts['cmeans'].set_color('green')
        violin_parts['cmedians'].set_color('blue')
        
        # Add optimal values
        opt_params = result_dict["opt_params"]
        plt.scatter(range(1, len(param_names)+1), opt_params, color='black', s=100, 
                   marker='*', label='Optimal Parameters')
        
        plt.xticks(range(1, len(param_names)+1), param_names)
        plt.ylabel('Parameter Value')
        plt.title('Distribution of Parameter Samples During Optimization')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'parameter_distribution.png'), dpi=300)
        plt.close()
    
    def visualize_spatial_error(self, predicted_series, actual_series, time_points, wm_data, slice_index):
        """Visualize spatial error between predicted and actual tumor for each time point."""
        if not predicted_series or not actual_series:
            print("Warning: No data available for spatial error visualization")
            return
            
        n_timepoints = len(time_points)
        fig, axes = plt.subplots(n_timepoints, 3, figsize=(15, 5*n_timepoints))
        
        # Handle single timepoint case
        if n_timepoints == 1:
            axes = np.array([axes])
        
        for t, (pred, actual, tp) in enumerate(zip(predicted_series, actual_series, time_points)):
            # Binary masks
            pred_bin = pred > 0.5
            actual_bin = actual > 0.5
            
            # Calculate error map
            # 0: Correct prediction (True Negative or True Positive)
            # 1: False Positive (predicted tumor where there isn't)
            # 2: False Negative (missed actual tumor)
            error_map = np.zeros_like(pred_bin, dtype=np.uint8)
            error_map[np.logical_and(pred_bin, ~actual_bin)] = 1  # False positive
            error_map[np.logical_and(~pred_bin, actual_bin)] = 2  # False negative
            
            # Create custom colormap: [correct, false positive, false negative]
            colors = ['black', 'red', 'blue']
            cmap = matplotlib.colors.ListedColormap(colors)
            
            # Plot actual tumor
            axes[t, 0].imshow(wm_data[:, :, slice_index], cmap='gray', alpha=1)
            axes[t, 0].imshow(actual_bin[:, :, slice_index], cmap='Greens', alpha=0.7)
            axes[t, 0].set_title(f"Actual - Time {tp}")
            axes[t, 0].axis('off')
            
            # Plot predicted tumor
            axes[t, 1].imshow(wm_data[:, :, slice_index], cmap='gray', alpha=1)
            axes[t, 1].imshow(pred_bin[:, :, slice_index], cmap='Oranges', alpha=0.7)
            axes[t, 1].set_title(f"Predicted - Time {tp}")
            axes[t, 1].axis('off')
            
            # Plot error map
            axes[t, 2].imshow(wm_data[:, :, slice_index], cmap='gray', alpha=0.5)
            axes[t, 2].imshow(error_map[:, :, slice_index], cmap=cmap, alpha=0.7, vmin=0, vmax=2)
            axes[t, 2].set_title(f"Error Map - Time {tp}")
            axes[t, 2].axis('off')
            
            # Add legend for the error map
            if t == 0:
                legend_elements = [
                    Line2D([0], [0], color='black', lw=4, label='Correct'),
                    Line2D([0], [0], color='red', lw=4, label='False Positive'),
                    Line2D([0], [0], color='blue', lw=4, label='False Negative')
                ]
                axes[t, 2].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'spatial_error_map.png'), dpi=300)
        plt.close()
    
    def visualize_volume_trajectory(self, predicted_series, actual_series, time_points, voxel_volume_mm3=1.0):
        """Compare tumor volume trajectories between predicted and actual data."""
        if not predicted_series or not actual_series:
            print("Warning: No data available for volume trajectory visualization")
            return
            
        # Calculate volumes
        actual_volumes = [np.sum(tumor > 0.5) * voxel_volume_mm3 for tumor in actual_series]
        predicted_volumes = [np.sum(tumor > 0.5) * voxel_volume_mm3 for tumor in predicted_series]
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, actual_volumes, 'bo-', linewidth=2, label='Actual Tumor', markersize=8)
        plt.plot(time_points, predicted_volumes, 'ro--', linewidth=2, label='Predicted Tumor', markersize=8)
        
        plt.xlabel('Time (weeks)', fontsize=14)
        plt.ylabel('Tumor Volume (mm³)', fontsize=14)
        plt.title('Tumor Volume Trajectory', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add volume labels
        for i, (actual, predicted) in enumerate(zip(actual_volumes, predicted_volumes)):
            plt.text(time_points[i], actual + max(actual_volumes)*0.02, f"{actual:.1f}", 
                    ha='center', va='bottom', color='blue', fontsize=10)
            plt.text(time_points[i], predicted - max(predicted_volumes)*0.05, f"{predicted:.1f}", 
                    ha='center', va='top', color='red', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'volume_trajectory.png'), dpi=300)
        plt.close()

    # Add to TimeSeriesLossVisualizer.py
    def visualize_train_test_performance(self, train_weeks, test_weeks, train_losses, test_losses):
        """Visualize performance on training and testing weeks."""
        plt.figure(figsize=(12, 6))
        
        # Plot training performance
        plt.plot(train_weeks, train_losses, 'bo-', linewidth=2, markersize=8, label='Training Loss')
        
        # Plot testing performance
        plt.plot(test_weeks, test_losses, 'ro--', linewidth=2, markersize=8, label='Testing Loss')
        
        plt.xlabel('Time (weeks)', fontsize=14)
        plt.ylabel('Loss Value', fontsize=14)
        plt.title('Model Performance on Training and Testing Data', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add loss values as text labels
        for week, loss in zip(train_weeks, train_losses):
            plt.text(week, loss + 0.02, f"{loss:.3f}", ha='center', va='bottom', color='blue', fontsize=10)
        
        for week, loss in zip(test_weeks, test_losses):
            plt.text(week, loss + 0.02, f"{loss:.3f}", ha='center', va='bottom', color='red', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'train_test_performance.png'), dpi=300)
        plt.close()