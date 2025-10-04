# TimeSeriesFittingSolver.py
import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
from scipy import ndimage
from TumorGrowthToolkit.FK import Solver as fwdSolver
import cmaes
import traceback
import re

class TimeSeriesFittingSolver:
    def __init__(self, settings, wm, gm, tumor_data, time_points):
        """
        Initialize the time series fitting solver.
        
        Args:
            settings (dict): Solver settings
            wm (numpy.ndarray): White matter data
            gm (numpy.ndarray): Grey matter data
            tumor_data (list): List of tumor segmentation masks at different time points
            time_points (list): List of corresponding time points (e.g., [0, 2, 5, 7] for weeks)
        """
        self.settings = settings
        self.wm = wm
        self.gm = gm
        self.tumor_data = tumor_data
        self.time_points = time_points
        
        # Initialize tracking variables
        self.all_evaluations = []  # Store all parameter sets and their losses
        self.best_params_history = []
        self.early_stopping = settings.get("early_stopping", True)
        self.patience = settings.get("patience", 5)
        
        # Validate inputs
        assert len(tumor_data) == len(time_points), "Number of tumor masks must match number of time points"
        assert len(time_points) >= 2, "At least two time points are required for trajectory fitting"
        
        # Sort time points and tumor data if not already sorted
        if not all(time_points[i] <= time_points[i+1] for i in range(len(time_points)-1)):
            sorted_indices = np.argsort(time_points)
            self.time_points = [time_points[i] for i in sorted_indices]
            self.tumor_data = [tumor_data[i] for i in sorted_indices]
        
        # Use the loss function specified in settings or default to soft dice
        self.loss_type = settings.get('loss_type', 'soft_dice')
        print(f"Using loss type: {self.loss_type}")

    def dice_coefficient(self, a, b):
        """Calculate the standard Dice similarity coefficient between two binary masks."""
        boolA, boolB = a > 0.5, b > 0.5  # Threshold to create binary masks
        if np.sum(boolA) + np.sum(boolB) == 0:
            return 0
        return 2 * np.sum(np.logical_and(boolA, boolB)) / (np.sum(boolA) + np.sum(boolB))

    def soft_dice_coefficient(self, a, b, margin=2):
        """Implementation of Dice with tolerance for small misalignments."""
        # Create binary masks
        boolA, boolB = a > 0.5, b > 0.5
        
        # If either mask is empty, return 0
        if np.sum(boolA) + np.sum(boolB) == 0:
            return 0
        
        # Dilate both masks slightly to allow overlap despite small misregistration
        struct = np.ones((margin, margin, margin))
        dilated_a = ndimage.binary_dilation(boolA, structure=struct)
        dilated_b = ndimage.binary_dilation(boolB, structure=struct)
        
        # Calculate standard Dice on original masks
        standard_dice = 2 * np.sum(np.logical_and(boolA, boolB)) / (np.sum(boolA) + np.sum(boolB))
        
        # Calculate Dice with dilated masks
        soft_dice_a = 2 * np.sum(np.logical_and(dilated_a, boolB)) / (np.sum(dilated_a) + np.sum(boolB))
        soft_dice_b = 2 * np.sum(np.logical_and(boolA, dilated_b)) / (np.sum(boolA) + np.sum(dilated_b))
        
        # Use the maximum of the two soft dice values
        soft_dice = max(soft_dice_a, soft_dice_b)
        
        # Weight between standard and soft Dice
        alpha = 0.7  # Adjustable weight
        return alpha * standard_dice + (1-alpha) * soft_dice

    def weighted_boundary_loss(self, a, b, sigma=3):
        """
        Weighted boundary loss that gives less weight to errors at tumor boundaries.
        
        Args:
            a: Predicted tumor mask
            b: Actual tumor mask
            sigma: Parameter controlling how quickly weight decreases near boundaries
            
        Returns:
            float: Loss value (0 to 1, where 0 is perfect match)
        """
        # Create binary masks
        boolA, boolB = a > 0.5, b > 0.5
        
        # If either mask is empty, return max loss
        if np.sum(boolA) == 0 or np.sum(boolB) == 0:
            return 1.0
        
        # Calculate distance transforms
        dist_a = ndimage.distance_transform_edt(boolA)
        dist_a_inv = ndimage.distance_transform_edt(~boolA)
        dist_b = ndimage.distance_transform_edt(boolB)
        dist_b_inv = ndimage.distance_transform_edt(~boolB)
        
        # Create boundary weight maps (weight is low near boundaries)
        boundary_a = np.exp(-(dist_a_inv * dist_a) / (2 * sigma**2))
        boundary_b = np.exp(-(dist_b_inv * dist_b) / (2 * sigma**2))
        
        # Combined boundary map (lower weight near either boundary)
        boundary_weight = np.minimum(boundary_a, boundary_b)
        
        # Calculate error map (absolute difference of masks)
        error_map = np.abs(boolA.astype(np.float32) - boolB.astype(np.float32))
        
        # Apply boundary weights to error
        weighted_error = error_map * (1 - 0.8 * boundary_weight)
        
        # Calculate weighted loss (mean of weighted error)
        loss = np.sum(weighted_error) / (np.sum(boolA) + np.sum(boolB))
        
        return loss

    def hausdorff_distance(self, a, b, percentile=95):
        """
        Calculate the modified Hausdorff distance between two binary masks.
        Uses a percentile instead of the maximum to make it more robust to outliers.
        
        Args:
            a: Predicted tumor mask
            b: Actual tumor mask
            percentile: Percentile to use instead of maximum (95 = 95th percentile)
            
        Returns:
            float: Normalized Hausdorff distance (0 to 1, where 0 is perfect match)
        """
        # Create binary masks
        boolA, boolB = a > 0.5, b > 0.5
        
        # If either mask is empty, return max distance
        if np.sum(boolA) == 0 or np.sum(boolB) == 0:
            return 1.0
        
        # Get coordinates of non-zero voxels
        coords_a = np.array(np.where(boolA)).T
        coords_b = np.array(np.where(boolB)).T
        
        # Calculate distances from each point in A to the nearest point in B
        distances_a_to_b = []
        for coord in coords_a:
            distances = np.sqrt(np.sum((coords_b - coord)**2, axis=1))
            distances_a_to_b.append(np.min(distances))
        
        # Calculate distances from each point in B to the nearest point in A
        distances_b_to_a = []
        for coord in coords_b:
            distances = np.sqrt(np.sum((coords_a - coord)**2, axis=1))
            distances_b_to_a.append(np.min(distances))
        
        # Calculate the percentile of distances in both directions
        if distances_a_to_b:
            percentile_a_to_b = np.percentile(distances_a_to_b, percentile)
        else:
            percentile_a_to_b = np.inf
            
        if distances_b_to_a:
            percentile_b_to_a = np.percentile(distances_b_to_a, percentile)
        else:
            percentile_b_to_a = np.inf
        
        # Modified Hausdorff is the maximum of the two percentiles
        hausdorff = max(percentile_a_to_b, percentile_b_to_a)
        
        # Normalize by the image diagonal to get a value between 0 and 1
        image_diagonal = np.sqrt(np.sum(np.array(a.shape)**2))
        normalized_hausdorff = min(hausdorff / image_diagonal, 1.0)
        
        return normalized_hausdorff

    def calculate_loss(self, pred, actual, loss_type=None):
        """
        Calculate loss between predicted and actual tumor masks using the specified loss type.
        
        Args:
            pred: Predicted tumor mask
            actual: Actual tumor mask
            loss_type: Type of loss to use (if None, use the default specified in settings)
            
        Returns:
            float: Loss value
        """
        if loss_type is None:
            loss_type = self.loss_type
        
        if loss_type == 'dice':
            # Standard Dice loss
            return 1 - self.dice_coefficient(pred, actual)
        elif loss_type == 'soft_dice':
            # Soft Dice with dilation for registration robustness
            return 1 - self.soft_dice_coefficient(pred, actual, 
                                               margin=self.settings.get('soft_dice_margin', 2))
        elif loss_type == 'boundary':
            # Boundary-weighted loss for registration robustness
            return self.weighted_boundary_loss(pred, actual, 
                                           sigma=self.settings.get('boundary_sigma', 3))
        elif loss_type == 'hausdorff':
            # Hausdorff distance loss for shape similarity
            return self.hausdorff_distance(pred, actual, 
                                       percentile=self.settings.get('hausdorff_percentile', 95))
        elif loss_type == 'combined':
            # Combined loss (weighted sum of multiple losses)
            dice_weight = self.settings.get('dice_weight', 0.4)
            soft_weight = self.settings.get('soft_weight', 0.3)
            boundary_weight = self.settings.get('boundary_weight', 0.2)
            hausdorff_weight = self.settings.get('hausdorff_weight', 0.1)
            
            dice_loss = 1 - self.dice_coefficient(pred, actual)
            soft_dice_loss = 1 - self.soft_dice_coefficient(pred, actual, 
                                                       margin=self.settings.get('soft_dice_margin', 2))
            boundary_loss = self.weighted_boundary_loss(pred, actual, 
                                                   sigma=self.settings.get('boundary_sigma', 3))
            hausdorff_loss = self.hausdorff_distance(pred, actual, 
                                               percentile=self.settings.get('hausdorff_percentile', 95))
            
            return (dice_weight * dice_loss + 
                    soft_weight * soft_dice_loss + 
                    boundary_weight * boundary_loss + 
                    hausdorff_weight * hausdorff_loss)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def time_series_loss(self, predicted_series, actual_series, time_weights=None):
        """
        Calculate loss between predicted and actual tumor series.
        
        Args:
            predicted_series (list): Predicted tumor masks at different time points
            actual_series (list): Actual tumor masks at different time points
            time_weights (list, optional): Weights for different time points
            
        Returns:
            tuple: (overall_loss, individual_losses)
        """
        if time_weights is None:
            # Equal weighting by default
            time_weights = [1.0] * len(actual_series)
        
        # Normalize weights
        total_weight = sum(time_weights)
        time_weights = [w / total_weight for w in time_weights]
        
        # Calculate loss for each time point
        individual_losses = []
        for pred, actual in zip(predicted_series, actual_series):
            loss = self.calculate_loss(pred, actual)
            individual_losses.append(loss)
        
        # Calculate weighted overall loss
        overall_loss = sum(loss * weight for loss, weight in zip(individual_losses, time_weights))
        
        return overall_loss, individual_losses

    def forward(self, x, resolution_factor=1.0):
        """
        Run forward simulation with given parameters.
        
        Args:
            x (list): Parameter vector [NxT1_pct, NyT1_pct, NzT1_pct, Dw, rho, t_start, time_scale]
            resolution_factor (float): Resolution factor for simulation
            
        Returns:
            list: Simulated tumor masks at each time point
        """
        # Extract standard parameters
        nx_pct, ny_pct, nz_pct, dw, rho = x[:5]
        
        # Extract time-related parameters (the new parameters we've added)
        t_start = x[5]  # Time before first observation when tumor started
        time_scale = x[6]  # Scaling factor between model and real time
        
        # Create adjusted time points based on time parameters
        adjusted_time_points = [(t - t_start) * time_scale for t in self.time_points]
        
        # Ensure all adjusted time points are positive
        if any(t < 0 for t in adjusted_time_points):
            # Invalid time parameters, tumor would start after first observation
            return None
        
        parameters = {
            'Dw': dw,               # Diffusion coefficient
            'rho': rho,              # Proliferation rate
            'RatioDw_Dg': 10,         # Ratio of diffusion coefficients
            'gm': self.gm,            # Grey matter data
            'wm': self.wm,            # White matter data
            'NxT1_pct': nx_pct,         # Initial tumor position
            'NyT1_pct': ny_pct,
            'NzT1_pct': nz_pct,
            'resolution_factor': resolution_factor,
            'segm': self.tumor_data[0],  # Initial tumor mask
            'time_points': adjusted_time_points,  # Adjusted time points
            't_start': t_start,          # New parameter: when tumor started
            'time_scale': time_scale,    # New parameter: time scaling factor
            'tumor_data': self.tumor_data,     # All tumor data for reference
            'real_time_unit': self.settings.get('real_time_unit', 'week'),
            'verbose': False,         # Disable verbose output during optimization
        }
        
        solver = fwdSolver(parameters)
        result = solver.solve()
        
        if not result['success']:
            # Return default high loss if simulation fails
            return None
            
        return result['time_series']

    def getLoss(self, x, gen=0):
        """
        Calculate loss for a parameter set.
        
        Args:
            x (list): Parameter vector
            gen (int): Current generation for adaptive resolution
            
        Returns:
            tuple: (loss_value, loss_details)
        """
        try:
            # Get worker ID for debugging
            worker_id = os.getpid()
            start_time = time.time()
            
            # Handle adaptive resolution if configured
            if isinstance(self.settings["resolution_factor"], dict):
                for relativeGen, resFactor in self.settings["resolution_factor"].items():
                    if gen / self.settings["generations"] >= relativeGen:
                        resolution_factor = resFactor
            elif isinstance(self.settings["resolution_factor"], float):
                resolution_factor = self.settings["resolution_factor"]
            else:
                raise ValueError("resolution_factor has to be float or dict")
            
            # Run forward model with parameters
            predicted_series = self.forward(x, resolution_factor)
            
            if predicted_series is None:
                # Simulation failed, return maximum loss
                loss = 1.0
                individual_losses = [1.0] * len(self.time_points)
                loss_details = {
                    "individual_losses": individual_losses,
                    "total_loss": loss,
                    "simulation_failed": True
                }
            else:
                # Calculate loss across time series
                time_weights = self.settings.get('time_weights', None)
                loss, individual_losses = self.time_series_loss(
                    predicted_series, 
                    self.tumor_data,
                    time_weights
                )
                
                loss_details = {
                    "individual_losses": individual_losses,
                    "total_loss": loss,
                    "simulation_failed": False,
                    "predicted_series": predicted_series
                }
        
            # Add execution information
            execution_time = time.time() - start_time
            loss_details["time"] = execution_time
            
            # Write parameter and loss to file for reliable tracking
            try:
                output_dir = self.settings.get("output_dir", ".")
                os.makedirs(output_dir, exist_ok=True)
                
                params_file = os.path.join(output_dir, "all_params.txt")
                with open(params_file, 'a') as f:
                    params_str = ", ".join([f"{p:.6f}" for p in x])
                    f.write(f"Loss: {loss:.6f}, Params: [{params_str}]\n")
            except Exception as e:
                print(f"Warning: Could not write to params file: {e}")
            
            print(f"Parameters: [{x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}, {x[3]:.3f}, {x[4]:.6f}, {x[5]:.2f}, {x[6]:.4f}], Loss: {loss:.4f}, Time: {execution_time:.2f}s")
            
            return loss, loss_details
            
        except Exception as e:
            # Catch any errors to prevent the whole optimization from crashing
            worker_id = os.getpid()  # Get current process ID in case it wasn't set
            print(f"Error in getLoss (Worker {worker_id}): {e}")
            print(traceback.format_exc())  # Print the full traceback
            
            # Return high loss value with error details
            return 1.0, {
                "total_loss": 1.0, 
                "error": str(e),
                "traceback": traceback.format_exc(),
                "worker_id": worker_id
            }

    def process_optimization_results(self, trace):
        """Process the optimization trace to extract results and track best parameters."""
        print("Processing optimization results...")
        
        # Find the best parameters from parameter file
        output_dir = self.settings.get("output_dir", ".")
        params_file = os.path.join(output_dir, "all_params.txt")
        
        best_loss = float('inf')
        best_params = None
        
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    try:
                        loss_match = re.search(r"Loss: ([\d\.]+)", line)
                        params_match = re.search(r"Params: \[([\d\.\-e, ]+)\]", line)
                        
                        if loss_match and params_match:
                            loss = float(loss_match.group(1))
                            params_str = params_match.group(1)
                            params = [float(p.strip()) for p in params_str.split(',')]
                            
                            if loss < best_loss:
                                best_loss = loss
                                best_params = params
                    except Exception as e:
                        print(f"Error parsing line: {e}")
                
                if best_params:
                    print(f"Found best parameters with loss {best_loss}")
                    print(f"Best parameters: {best_params}")
                else:
                    print("WARNING: No valid parameters found in params file.")
            except Exception as e:
                print(f"Error reading params file: {e}")
        else:
            print(f"WARNING: Params file {params_file} not found.")
        
        # If we didn't find any parameters, use defaults to avoid crashing
        if best_params is None:
            best_loss = 0.5  # Arbitrary value
            best_params = [0.5, 0.5, 0.5, 1.0, 0.06, -10.0, 1.0]
            print("WARNING: Using default parameters.")
        
        # Run forward simulation with best parameters
        best_predicted = None
        if best_params is not None:
            try:
                print("Running forward model with best parameters...")
                best_predicted = self.forward(best_params)
            except Exception as e:
                print(f"Cannot run forward model: {e}")
        
        # Create result dictionary
        result_dict = {
            "minLoss": best_loss,
            "opt_params": best_params,
            "best_predicted": best_predicted,
            "all_evaluations": []  # Empty for now
        }
        
        # Save best parameters to file
        try:
            result_file = os.path.join(output_dir, "best_params_results.npy")
            np.save(result_file, {
                "opt_params": best_params,
                "minLoss": best_loss,
                "time_min": 0,  # Will be updated later
                "time_sec": 0   # Will be updated later
            })
            print(f"Saved best parameters to {result_file}")
        except Exception as e:
            print(f"Error saving result file: {e}")
        
        return result_dict

    def run(self):
        """
        Run the optimization to find the best parameters.
        
        Returns:
            tuple: (best_predicted_timeseries, result_dictionary)
        """
        start_time = time.time()
        
        # Reset tracking variables for a fresh run
        self.best_params_history = []
        self.all_evaluations = []

        # Initial parameter values (now including t_start and time_scale)
        initial_params = [
            self.settings.get("NxT1_pct0", 0.5),
            self.settings.get("NyT1_pct0", 0.5),
            self.settings.get("NzT1_pct0", 0.5),
            self.settings.get("dw0", 1.0),
            self.settings.get("rho0", 0.06),
            self.settings.get("t_start0", -10.0),  # Default: tumor started 10 weeks before first observation
            self.settings.get("time_scale0", 1.0)  # Default: no scaling between model and real time
        ]
        
        # Parameter ranges for optimization (now including t_start and time_scale)
        param_ranges = self.settings.get("parameterRanges", [
            [0, 1],           # NxT1_pct
            [0, 1],           # NyT1_pct
            [0, 1],           # NzT1_pct
            [0.001, 3],       # Dw
            [0.0001, 0.225],  # rho
            [-100, 0],        # t_start (tumor can start up to 100 weeks before first observation)
            [0.1, 10]         # time_scale (model time can be 0.1x to 10x real time)
        ])
        
        # Create/clear the parameter tracking file
        try:
            output_dir = self.settings.get("output_dir", ".")
            os.makedirs(output_dir, exist_ok=True)
            params_file = os.path.join(output_dir, "all_params.txt")
            with open(params_file, 'w') as f:
                f.write("# Parameters and losses from optimization\n")
        except Exception as e:
            print(f"Warning: Could not create params file: {e}")
        
        # Run standard CMA-ES optimization with error handling
        try:
            print("Starting CMA-ES optimization...")
            trace = cmaes.cmaes(
                self.getLoss,
                initial_params,
                self.settings["sigma0"],
                self.settings["generations"],
                workers=self.settings.get("workers", 1),
                trace=True,
                parameterRange=param_ranges
            )
            print("CMA-ES optimization completed")
            
            # Process optimization results and track best parameters
            result_dict = self.process_optimization_results(trace)
    
            # Get best predicted from results
            best_predicted = result_dict.get("best_predicted", None)
    
            # Add execution time
            result_dict["time_min"] = (time.time() - start_time) / 60
            result_dict["time_sec"] = time.time() - start_time
            
            # Update the results file with timing information
            try:
                output_dir = self.settings.get("output_dir", ".")
                result_file = os.path.join(output_dir, "best_params_results.npy")
                if os.path.exists(result_file):
                    results = np.load(result_file, allow_pickle=True).item()
                    results["time_min"] = result_dict["time_min"]
                    results["time_sec"] = result_dict["time_sec"]
                    np.save(result_file, results)
            except Exception as e:
                print(f"Could not update timing in results file: {e}")
            
            return best_predicted, result_dict
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            print(traceback.format_exc())  # Print the full traceback
            
            # Return minimal result dictionary
            result_dict = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "time_min": (time.time() - start_time) / 60,
                "time_sec": time.time() - start_time,
                "early_termination": True
            }
            return None, result_dict