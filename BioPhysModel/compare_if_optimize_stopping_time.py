import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from scipy import ndimage
import cmaesFK
from TumorGrowthToolkit.FK import Solver as fwdSolver
import cmaes
import tools

# Set matplotlib to use Agg backend (non-interactive, file-only)
import matplotlib
matplotlib.use('Agg')

class ModelComparisonPipeline:
    def __init__(self, data_path, results_path, num_runs=3, 
                use_early_stopping=True, patience=4, min_improvement=0.015):
        self.data_path = data_path
        self.results_path = results_path
        self.num_runs = num_runs
        self.use_early_stopping = use_early_stopping
        self.patience = patience
        self.min_improvement = min_improvement
        os.makedirs(results_path, exist_ok=True)
        
        # Create a plots directory
        self.plots_dir = os.path.join(results_path, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Load data once
        self.WM, self.GM, self.segmentation = self.load_data()
        self.edema = np.logical_or(self.segmentation == 3, self.segmentation == 2)
        self.necrotic = self.segmentation == 1
        self.enhancing = self.segmentation == 4
        
        # Results storage
        self.results_fixed = []
        self.results_optimized = []
    
    def load_data(self):
        """Load data for tumor growth modeling"""
        import nibabel as nib
        
        wm_path = os.path.join(self.data_path, 'T1_pve_1.nii.gz')
        gm_path = os.path.join(self.data_path, 'T1_pve_2.nii.gz')
        tumor_path = os.path.join(self.data_path, 't1_seg_mask.nii.gz')
        
        WM = nib.load(wm_path).get_fdata()
        GM = nib.load(gm_path).get_fdata()
        segmentation = nib.load(tumor_path).get_fdata()
        
        return WM, GM, segmentation
    
    def initialize_settings(self, include_stopping_time=False):
        """Initialize solver settings with or without stopping time as parameter"""
        settings = {}
        com = ndimage.center_of_mass(self.edema)
        
        # Initial parameters
        settings["rho0"] = 0.06
        settings["dw0"] = 1.0
        settings["thresholdT1c"] = 0.675
        settings["thresholdFlair"] = 0.25
        settings["NxT1_pct0"] = float(com[0] / self.GM.shape[0])
        settings["NyT1_pct0"] = float(com[1] / self.GM.shape[1])
        settings["NzT1_pct0"] = float(com[2] / self.GM.shape[2])
        
        if include_stopping_time:
            settings["stopping_time0"] = 50  # Initial value for stopping time
        
        # Parameter ranges - add stopping_time range if needed
        if include_stopping_time:
            settings["parameterRanges"] = [
                [0, 1], [0, 1], [0, 1],                # Position
                [0.001, 3], [0.0001, 0.225],           # Diffusion and proliferation
                [10, 200],                             # Stopping time
                [0.5, 0.85], [0.001, 0.5]              # Thresholds
            ]
        else:
            settings["parameterRanges"] = [
                [0, 1], [0, 1], [0, 1],                # Position
                [0.001, 3], [0.0001, 0.225],           # Diffusion and proliferation
                [0.5, 0.85], [0.001, 0.5]              # Thresholds
            ]
        
        # Optimization settings
        settings["workers"] = 9
        settings["sigma0"] = 0.02
        settings["resolution_factor"] = {0: 0.3, 0.7: 0.5}
        settings["generations"] = 12
        
        return settings
    
    def run_fixed_model(self, run_id):
        """Run optimization with fixed stopping time"""
        settings = self.initialize_settings(include_stopping_time=False)
        
        # Original solver

        # Initialize CMA-ES solver

        solver = cmaesFK.CmaesSolver(settings, self.WM, self.GM, 
                                  self.edema, self.enhancing, self.necrotic, self.segmentation)
        
        # Run optimization
        start_time = time.time()
        resultTumor, resultDict = solver.run(early_stopping=True, patience=4, min_improvement=0.015)
        end_time = time.time()
        
        # Store results with fixed stopping time of 100
        params = resultDict['opt_params']
        run_results = {
            'run_id': run_id,
            'model_type': 'fixed',
            'min_loss': resultDict['minLoss'],
            'runtime_min': (end_time - start_time) / 60,
            'params': {
                'Dw': params[3],
                'rho': params[4],
                'NxT1_pct': params[0],
                'NyT1_pct': params[1],
                'NzT1_pct': params[2],
                'stopping_time': 100,  # Fixed value
                'thresholdT1c': params[-2],
                'thresholdFlair': params[-1]
            },
            'resultDict': resultDict
        }
        
        self.results_fixed.append(run_results)
        return run_results
    
    def run_optimized_model(self, run_id):
        """Run optimization with optimizable stopping time"""
        settings = self.initialize_settings(include_stopping_time=True)
        
        # Modified solver with stopping time parameter
        solver = ModifiedCmaesSolver(settings, self.WM, self.GM, 
                                  self.edema, self.enhancing, self.necrotic, self.segmentation)
        
        start_time = time.time()
        resultTumor, resultDict = solver.run(early_stopping=True, patience=4, min_improvement=0.015)
        end_time = time.time()
        
        # Store results with optimized stopping time
        params = resultDict['opt_params']
        run_results = {
            'run_id': run_id,
            'model_type': 'optimized',
            'min_loss': resultDict['minLoss'],
            'runtime_min': (end_time - start_time) / 60,
            'params': {
                'Dw': params[3],
                'rho': params[4],
                'NxT1_pct': params[0],
                'NyT1_pct': params[1],
                'NzT1_pct': params[2],
                'stopping_time': params[5],  # Optimized value
                'thresholdT1c': params[-2],
                'thresholdFlair': params[-1]
            },
            'resultDict': resultDict
        }
        
        self.results_optimized.append(run_results)
        return run_results
    
    def compare_models(self):
        """Run full comparison between fixed and optimized models"""
        print("Starting model comparison...")
        
        for i in range(self.num_runs):
            print(f"Run {i+1}/{self.num_runs}")
            
            print("  Running fixed stopping time model...")
            self.run_fixed_model(i)
            
            print("  Running optimized stopping time model...")
            self.run_optimized_model(i)
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze and visualize comparison results"""
        # Create summary dataframe
        all_results = []
        for r in self.results_fixed + self.results_optimized:
            result = {
                'Run': r['run_id'],
                'Model': r['model_type'],
                'Min Loss': r['min_loss'],
                'Runtime (min)': r['runtime_min']
            }
            result.update({k: v for k, v in r['params'].items()})
            all_results.append(result)
            
        df = pd.DataFrame(all_results)
        
        # Summary statistics
        summary = df.groupby('Model').agg({
            'Min Loss': ['mean', 'std', 'min'],
            'Runtime (min)': ['mean', 'std'],
            'Dw': ['mean', 'std'],
            'rho': ['mean', 'std'],
            'stopping_time': ['mean', 'std']
        })
        
        # Save results
        df.to_csv(os.path.join(self.results_path, 'model_comparison_results.csv'))
        summary.to_csv(os.path.join(self.results_path, 'model_comparison_summary.csv'))
        
        # Generate visualizations
        self.plot_loss_comparison(df)
        self.plot_parameter_comparison(df)
        self.analyze_stopping_criteria()
        
        return summary
    
    def plot_loss_comparison(self, df):
        """Plot loss comparison between models"""
        plt.figure(figsize=(12, 6))
        
        # Model performance comparison
        plt.subplot(1, 2, 1)
        fixed_loss = df[df['Model'] == 'fixed']['Min Loss']
        opt_loss = df[df['Model'] == 'optimized']['Min Loss']
        
        plt.bar([0, 1], [fixed_loss.mean(), opt_loss.mean()], 
                yerr=[fixed_loss.std(), opt_loss.std()],
                capsize=10)
        plt.xticks([0, 1], ['Fixed', 'Optimized'])
        plt.ylabel('Loss (lower is better)')
        plt.title('Model Performance Comparison')
        
        # Runtime comparison
        plt.subplot(1, 2, 2)
        fixed_time = df[df['Model'] == 'fixed']['Runtime (min)']
        opt_time = df[df['Model'] == 'optimized']['Runtime (min)']
        
        plt.bar([0, 1], [fixed_time.mean(), opt_time.mean()],
                yerr=[fixed_time.std(), opt_time.std()],
                capsize=10)
        plt.xticks([0, 1], ['Fixed', 'Optimized'])
        plt.ylabel('Runtime (minutes)')
        plt.title('Computational Cost Comparison')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'loss_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Loss comparison plot saved to: {save_path}")
    
    def plot_parameter_comparison(self, df):
        """Plot comparison of optimized parameters"""
        plt.figure(figsize=(15, 8))
        
        params = ['Dw', 'rho', 'stopping_time', 'NxT1_pct', 'NyT1_pct', 'NzT1_pct']
        
        for i, param in enumerate(params):
            plt.subplot(2, 3, i+1)
            
            fixed_vals = df[df['Model'] == 'fixed'][param]
            opt_vals = df[df['Model'] == 'optimized'][param]
            
            plt.boxplot([fixed_vals, opt_vals], labels=['Fixed', 'Optimized'])
            plt.ylabel(param)
            plt.title(f'{param} Distribution')
            
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'parameter_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Parameter comparison plot saved to: {save_path}")
    
    def analyze_stopping_criteria(self):
        """Analyze the optimization stopping criteria effectiveness"""
        plt.figure(figsize=(12, 8))
        
        # Extract loss histories for analysis
        fixed_histories = []
        for res in self.results_fixed:
            history = []
            for gen in res['resultDict']['lossDir']:
                for run in gen:
                    history.append(run['lossTotal'])
            fixed_histories.append(history)
        
        opt_histories = []
        for res in self.results_optimized:
            history = []
            for gen in res['resultDict']['lossDir']:
                for run in gen:
                    history.append(run['lossTotal'])
            opt_histories.append(history)
        
        # Plot convergence curves
        plt.subplot(2, 1, 1)
        for history in fixed_histories:
            plt.plot(history, 'b-', alpha=0.3)
        for history in opt_histories:
            plt.plot(history, 'r-', alpha=0.3)
            
        # Add mean convergence curves
        if fixed_histories:
            mean_fixed = np.mean([h[:min(len(h) for h in fixed_histories)] for h in fixed_histories], axis=0)
            plt.plot(mean_fixed, 'b-', linewidth=2, label='Fixed Stopping Time')
            
        if opt_histories:
            mean_opt = np.mean([h[:min(len(h) for h in opt_histories)] for h in opt_histories], axis=0)
            plt.plot(mean_opt, 'r-', linewidth=2, label='Optimized Stopping Time')
            
        plt.xlabel('Evaluations')
        plt.ylabel('Loss')
        plt.title('Convergence Comparison')
        plt.legend()
        
        # Analyze if optimization terminated too early
        plt.subplot(2, 1, 2)
        
        # Calculate improvement in last 20% of evaluations
        improvements = []
        for model, histories in [('Fixed', fixed_histories), ('Optimized', opt_histories)]:
            model_improvements = []
            for history in histories:
                cutoff = int(0.8 * len(history))
                if cutoff < len(history):
                    start_loss = history[cutoff]
                    final_loss = history[-1]
                    improvement = (start_loss - final_loss) / start_loss * 100
                    model_improvements.append(improvement)
            improvements.append(np.mean(model_improvements) if model_improvements else 0)
        
        plt.bar(['Fixed', 'Optimized'], improvements)
        plt.ylabel('% Improvement in Final 20% of Evaluations')
        plt.title('Assessment of Optimization Termination')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'stopping_criteria_analysis.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Stopping criteria analysis plot saved to: {save_path}")


class ModifiedCmaesSolver(cmaesFK.CmaesSolver):
    """CmaesSolver extended to optimize stopping time parameter"""
    
    def forward(self, x, resolution_factor=1.0):
        """Forward model with stopping time as an optimizable parameter"""
        parameters = {
            'Dw': x[3],               # Diffusion coefficient for white matter
            'rho': x[4],              # Proliferation rate
            'stopping_time': x[5],    # Optimizable stopping time parameter
            'RatioDw_Dg': 10,         # Ratio of diffusion coefficients
            'gm': self.gm,            # Grey matter data
            'wm': self.wm,            # White matter data
            'NxT1_pct': x[0],         # Tumor position (percentages)
            'NyT1_pct': x[1],
            'NzT1_pct': x[2],
            'resolution_factor': resolution_factor,
            'segm': self.segm
        }
        print("run: ", x)
        solver = fwdSolver(parameters)
        return solver.solve()["final_state"]
    
    def run(self, early_stopping=False, patience=5, min_improvement=0.01):
        start = time.time()
        
        initValues = (self.settings["NxT1_pct0"], self.settings["NyT1_pct0"], self.settings["NzT1_pct0"], 
                    self.settings["dw0"], self.settings["rho0"], 
                    self.settings.get("stopping_time0", 100),  # Add default in case not present
                    self.settings["thresholdT1c"], self.settings["thresholdFlair"])

        trace = cmaes.cmaes(self.getLoss, initValues, self.settings["sigma0"], 
                        self.settings["generations"], workers=self.settings["workers"], 
                        trace=True, parameterRange=self.settings["parameterRanges"],
                        early_stopping=early_stopping, patience=patience, 
                        min_improvement=min_improvement)



        trace = cmaes.cmaes(self.getLoss, initValues, self.settings["sigma0"], 
                           self.settings["generations"], workers=self.settings["workers"], 
                           trace=True, parameterRange=self.settings["parameterRanges"])
        #trace = np.array(trace)
        nsamples, y0s, xs0s, sigmas, Cs, pss, pcs, Cmus, C1s, xmeans, lossDir = [], [], [], [], [], [], [], [], [], [], []
        for element in trace:
            nsamples.append(element[0])
            y0s.append(element[1])
            xs0s.append(element[2])
            sigmas.append(element[3])
            Cs.append(element[4])
            pss.append(element[5])
            pcs.append(element[6])
            Cmus.append(element[7])
            C1s.append(element[8])
            xmeans.append(element[9])
            lossDir.append(element[10])

        
        minLoss = 1
        for i in range(len(lossDir)):
            for j in range(len(lossDir[i])):
                if lossDir[i][j]["lossTotal"] <= minLoss:
                    minLoss = lossDir[i][j]["lossTotal"]
                    opt = lossDir[i][j]["allParams"]

        tumor = self.forward(opt)
        end = time.time()

        resultDict = {}

        resultDict["nsamples"] = nsamples
        resultDict["y0s"] = y0s
        resultDict["xs0s"] = xs0s
        resultDict["sigmas"] = sigmas
        resultDict["Cs"] = Cs
        resultDict["pss"] = pss
        resultDict["pcs"] = pcs
        resultDict["Cmus"] = Cmus
        resultDict["C1s"] = C1s
        resultDict["xmeans"] = xmeans
        resultDict["lossDir"] = lossDir
        resultDict["minLoss"] = minLoss
        resultDict["opt_params"] = opt
        resultDict["time_min"] = (end - start) / 60
        
        return tumor, resultDict


if __name__ == "__main__":
    # Define paths
    data_path = '/scratch/bcog/jyin15/Tumor/BioPhysModel/parcelled/Patient-091/week-000'
    results_path = '/scratch/bcog/jyin15/Tumor/BioPhysModel/model_comparison'

    # Run comparison
    pipeline = ModelComparisonPipeline(
    data_path, 
    results_path, 
    num_runs=3,
    use_early_stopping=True, 
    patience=4, 
    min_improvement=0.015
    )
    summary = pipeline.compare_models()
    print("Comparison complete. Results saved to:", results_path)