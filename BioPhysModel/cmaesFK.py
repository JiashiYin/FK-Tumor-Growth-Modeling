from TumorGrowthToolkit.FK import Solver as fwdSolver
import cmaes
import numpy as np
import nibabel as nib
import time
import os
        
def dice(a, b):
    boolA, boolB = a > 0, b > 0 
    if np.sum(boolA) + np.sum(boolB) == 0:
        return 0

    return 2 * np.sum( np.logical_and(boolA, boolB)) / (np.sum(boolA) + np.sum(boolB))

class CmaesSolver():
    def __init__(self,  settings, wm, gm, edema, enhancing, necrotic, segm):
        self.settings = settings
        self.wm = wm
        self.gm = gm
        self.edema = edema
        self.enhancing = enhancing
        self.necrotic = necrotic
        self.segm = segm
        
        # For tracking best parameters
        self.best_loss = float('inf')
        self.best_params = None
        self.all_evaluations = []


    def lossfunction(self, tumor, thresholdT1c, thresholdFlair):
        lambdaFlair = 0.333
        lambdaT1c = 0.333
        
        proposedEdema = np.logical_and(tumor > thresholdFlair, tumor < thresholdT1c)
        lossFlair = 1 - dice(proposedEdema, self.edema)
        lossT1c = 1 - dice(tumor > thresholdT1c, np.logical_or(self.necrotic, self.enhancing))
        loss = lambdaFlair * lossFlair + lambdaT1c * lossT1c

        # Handle invalid loss values
        if not loss <= 1:
            loss = 1

        return loss, {"lossFlair": lossFlair, "lossT1c": lossT1c, "lossTotal": loss}


    def forward(self, x, resolution_factor=1.0):
        # Include model_time parameter (x[5])
        parameters = {
            'Dw': x[3],         # Diffusion coefficient for white matter from dw0
            'rho': x[4],        # Proliferation rate from rho0
            'stopping_time': x[5],  # Model time parameter (new)
            'RatioDw_Dg': 10,   # Ratio of diffusion coefficients in white and grey matter
            'gm': self.gm,      # Grey matter data
            'wm': self.wm,      # White matter data
            'NxT1_pct': x[0],   # Initial focal position (in percentages)
            'NyT1_pct': x[1],
            'NzT1_pct': x[2],
            'resolution_factor': resolution_factor,
            'segm': self.segm
        }
        
        print(f"Running: Dw={x[3]:.4f}, rho={x[4]:.6f}, time={x[5]:.1f}")
        
        solver = fwdSolver(parameters)
        result = solver.solve()
        
        if not result.get('success', False):
            print(f"Forward model failed: {result.get('error', 'Unknown error')}")
            return None
            
        return result["final_state"]


    def getLoss(self, x, gen):
        start_time = time.time()

        # Handle adaptive resolution
        if isinstance(self.settings["resolution_factor"], dict):
            resolution_factor = None
            for relativeGen, resFactor in self.settings["resolution_factor"].items():
                if gen / self.settings["generations"] >= relativeGen:
                    resolution_factor = resFactor
        elif isinstance(self.settings["resolution_factor"], float):
            resolution_factor = self.settings["resolution_factor"]
        else:
            raise ValueError("resolution_factor has to be float or dict")
        
        # Run forward model
        tumor = self.forward(x[:-2], resolution_factor)
        
        if tumor is None:
            # Handle failed simulation
            loss = 1.0
            lossDir = {
                "lossFlair": 1.0,
                "lossT1c": 1.0, 
                "lossTotal": 1.0,
                "simulation_failed": True
            }
        else:
            # Calculate loss
            thresholdT1c = x[-2]    
            thresholdFlair = x[-1]
            loss, lossDir = self.lossfunction(tumor, thresholdT1c, thresholdFlair)
        
        # Record execution details
        end_time = time.time()
        lossDir["time"] = end_time - start_time
        lossDir["allParams"] = x.copy()
        lossDir["resolution_factor"] = resolution_factor
        
        # Store all evaluations for later analysis
        self.all_evaluations.append((loss, x.copy(), lossDir))
        
        # Track best parameters
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = x.copy()
            print(f"New best loss: {loss:.6f}")
        
        print(f"Loss: {loss:.6f}, Time: {end_time - start_time:.2f}s")
        
        return loss, lossDir

    def run(self):
        start = time.time()
        
        # Include model_time parameter in initial values
        initValues = (
            self.settings["NxT1_pct0"], 
            self.settings["NyT1_pct0"], 
            self.settings["NzT1_pct0"], 
            self.settings["dw0"], 
            self.settings["rho0"],
            self.settings["model_time0"],  # New parameter 
            self.settings["thresholdT1c"], 
            self.settings["thresholdFlair"]
        )

        # Run CMA-ES optimization
        trace = cmaes.cmaes(
            self.getLoss, 
            initValues, 
            self.settings["sigma0"], 
            self.settings["generations"], 
            workers=self.settings["workers"], 
            trace=True, 
            parameterRange=self.settings["parameterRanges"]
        )

        # Find best parameters from all evaluations
        min_loss_entry = min(self.all_evaluations, key=lambda x: x[0])
        min_loss = min_loss_entry[0]
        opt_params = min_loss_entry[1]
        
        print(f"Best parameters found with loss: {min_loss:.6f}")
        
        # Run forward model with best parameters
        tumor = self.forward(opt_params)
        end = time.time()

        # Prepare result dictionary
        resultDict = {
            "trace": trace,
            "all_evaluations": self.all_evaluations,
            "minLoss": min_loss,
            "opt_params": opt_params,
            "time_min": (end - start) / 60
        }
        
        return tumor, resultDict