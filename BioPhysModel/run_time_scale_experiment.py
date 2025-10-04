#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import time
import glob

# Configuration
PATIENT_ID = "Patient-042"
BASE_DIR = "/scratch/bcsl/jyin15/Tumor/BioPhysModel"
DATA_DIR = f"{BASE_DIR}/parcelled_and_masks"
OUTPUT_ROOT = f"{BASE_DIR}/time_scale_experiment"
PIPELINE_SCRIPT = f"{BASE_DIR}/TumorGrowthPipeline.py"
NUM_WORKERS = 8
FK_GENERATIONS = 15
SCALE_GENERATIONS = 15

# Define the experiment time point pairs
TIME_POINT_PAIRS = [
    (4, 22),   # Pair 1: weeks 4-22
    (22, 49),  # Pair 2: weeks 22-49
    (49, 60),  # Pair 3: weeks 49-60
    (60, 81)   # Pair 4: weeks 60-81
]

def run_experiment():
    """Run the experiment for all time point pairs"""
    # Create the output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Store results
    results = {
        'patient_id': PATIENT_ID,
        'time_points': [],
        'time_diffs': [],
        'time_scales': [],
        'fk_dice': [],
        'scale_dice': [],
        'output_dirs': []
    }
    
    # Run the pipeline for each time point pair
    for i, (fit_time, scale_time) in enumerate(TIME_POINT_PAIRS):
        print(f"\n{'='*80}")
        print(f"Running experiment {i+1}/4: Weeks {fit_time} → {scale_time}")
        print(f"{'='*80}\n")
        
        # Create a unique output directory for this run
        output_dir = f"{OUTPUT_ROOT}/{PATIENT_ID}_weeks_{fit_time}_{scale_time}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the command
        cmd = [
            "python", PIPELINE_SCRIPT,
            "--data_dir", DATA_DIR,
            "--patient_id", PATIENT_ID,
            "--output_dir", output_dir,
            "--fit_time", str(fit_time),
            "--scale_time", str(scale_time),
            "--test_time", str(scale_time),  # Use same as scale_time to skip actual testing
            "--fk_generations", str(FK_GENERATIONS),
            "--scale_generations", str(SCALE_GENERATIONS),
            "--workers", str(NUM_WORKERS)
        ]
        
        # Run the command
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Save the output
        with open(f"{output_dir}/pipeline_output.log", "w") as f:
            f.write(process.stdout)
        
        with open(f"{output_dir}/pipeline_error.log", "w") as f:
            f.write(process.stderr)
        
        print(f"Pipeline completed for weeks {fit_time} → {scale_time}")
        
        # Extract the time scale parameter from the results
        time_scale = extract_time_scale(output_dir)
        time_diff = scale_time - fit_time
        
        # Extract dice scores
        fk_dice = extract_dice_score(output_dir, 'fk')
        scale_dice = extract_dice_score(output_dir, 'scale')
        
        # Store the results
        results['time_points'].append(f"{fit_time} → {scale_time}")
        results['time_diffs'].append(time_diff)
        results['time_scales'].append(time_scale)
        results['fk_dice'].append(fk_dice)
        results['scale_dice'].append(scale_dice)
        results['output_dirs'].append(output_dir)
        
        print(f"Time scale for weeks {fit_time} → {scale_time}: {time_scale}")
        print(f"FK Dice score: {fk_dice}")
        print(f"Scale Dice score: {scale_dice}")
    
    return results

def extract_time_scale(output_dir):
    """Extract the time scale parameter from the results"""
    try:
        # Look for the scale json file
        scale_results_file = glob.glob(f"{output_dir}/scale/*/results.json")
        if scale_results_file:
            with open(scale_results_file[0], "r") as f:
                results = json.load(f)
                return results.get("time_scale", None)
    except Exception as e:
        print(f"Error extracting time scale: {e}")
    
    return None

def extract_dice_score(output_dir, phase):
    """Extract the dice score from the results"""
    try:
        # Look for the results json file
        results_file = glob.glob(f"{output_dir}/{phase}/*/results.json")
        if results_file:
            with open(results_file[0], "r") as f:
                results = json.load(f)
                return results.get("dice", None)
    except Exception as e:
        print(f"Error extracting dice score: {e}")
    
    return None

def visualize_time_scales(results):
    """Create visualizations of the time scales"""
    # Create output directory for visualizations
    viz_dir = f"{OUTPUT_ROOT}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Bar chart of time scales
    plt.figure(figsize=(12, 8))
    plt.bar(results['time_points'], results['time_scales'], color='steelblue')
    plt.xlabel('Time Point Pair (weeks)', fontsize=12)
    plt.ylabel('Time Scale Parameter', fontsize=12)
    plt.title(f'Time Scale Stability Analysis - {PATIENT_ID}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels
    for i, v in enumerate(results['time_scales']):
        if v is not None:
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/time_scale_comparison.png", dpi=300)
    plt.close()
    
    # 2. Time scale vs. time difference
    plt.figure(figsize=(10, 6))
    plt.scatter(results['time_diffs'], results['time_scales'], s=100, color='darkblue')
    
    # Add regression line if we have enough data points
    if all(x is not None for x in results['time_scales']):
        z = np.polyfit(results['time_diffs'], results['time_scales'], 1)
        p = np.poly1d(z)
        plt.plot(results['time_diffs'], p(results['time_diffs']), "r--", linewidth=2)
        plt.text(0.05, 0.95, f'y = {z[0]:.6f}x + {z[1]:.4f}', 
                 transform=plt.gca().transAxes, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8))
    
    # Add labels to each point
    for i, (x, y, label) in enumerate(zip(results['time_diffs'], results['time_scales'], results['time_points'])):
        plt.annotate(
            label, (x, y),
            xytext=(10, 5), textcoords='offset points',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.xlabel('Time Difference (weeks)', fontsize=12)
    plt.ylabel('Time Scale Parameter', fontsize=12)
    plt.title(f'Time Scale vs. Time Difference - {PATIENT_ID}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/time_scale_vs_difference.png", dpi=300)
    plt.close()
    
    # 3. Relation between dice scores and time scale
    plt.figure(figsize=(12, 6))
    
    # Plot FK dice vs time scale
    plt.subplot(1, 2, 1)
    plt.scatter(results['time_scales'], results['fk_dice'], s=80, color='blue', label='FK Dice')
    
    for i, (x, y, label) in enumerate(zip(results['time_scales'], results['fk_dice'], results['time_points'])):
        plt.annotate(
            label, (x, y),
            xytext=(10, 5), textcoords='offset points',
            fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
        )
    
    plt.xlabel('Time Scale', fontsize=12)
    plt.ylabel('FK Dice Score', fontsize=12)
    plt.title('FK Dice vs Time Scale', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Scale dice vs time scale
    plt.subplot(1, 2, 2)
    plt.scatter(results['time_scales'], results['scale_dice'], s=80, color='green', label='Scale Dice')
    
    for i, (x, y, label) in enumerate(zip(results['time_scales'], results['scale_dice'], results['time_points'])):
        plt.annotate(
            label, (x, y),
            xytext=(10, 5), textcoords='offset points',
            fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
        )
    
    plt.xlabel('Time Scale', fontsize=12)
    plt.ylabel('Scale Dice Score', fontsize=12)
    plt.title('Scale Dice vs Time Scale', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/dice_vs_time_scale.png", dpi=300)
    plt.close()
    
    return viz_dir

def create_summary_report(results, viz_dir):
    """Create a comprehensive summary report"""
    report_dir = f"{OUTPUT_ROOT}/reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # Create an HTML report
    html_path = f"{report_dir}/{PATIENT_ID}_time_scale_analysis.html"
    
    with open(html_path, 'w') as f:
        # Write HTML header
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Time Scale Stability Analysis - {PATIENT_ID}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric {{ font-weight: bold; color: #2980b9; }}
                .section {{ margin-top: 30px; border-left: 5px solid #3498db; padding-left: 15px; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .container {{ display: flex; flex-wrap: wrap; }}
                .image-container {{ margin: 10px; }}
            </style>
        </head>
        <body>
            <h1>Time Scale Stability Analysis</h1>
            <h2>Patient: {PATIENT_ID}</h2>
            <p>Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        ''')
        
        # Summary section
        f.write('''
            <div class="section">
                <h2>Summary of Results</h2>
                <table>
                    <tr>
                        <th>Time Point Pair</th>
                        <th>Time Difference (weeks)</th>
                        <th>Time Scale Parameter</th>
                        <th>FK Dice Score</th>
                        <th>Scale Dice Score</th>
                    </tr>
        ''')
        
        for i in range(len(results['time_points'])):
            f.write(f'''
                    <tr>
                        <td>{results['time_points'][i]}</td>
                        <td>{results['time_diffs'][i]}</td>
                        <td class="metric">{results['time_scales'][i]:.4f if results['time_scales'][i] is not None else 'N/A'}</td>
                        <td>{results['fk_dice'][i]:.4f if results['fk_dice'][i] is not None else 'N/A'}</td>
                        <td>{results['scale_dice'][i]:.4f if results['scale_dice'][i] is not None else 'N/A'}</td>
                    </tr>
            ''')
        
        f.write('''
                </table>
            </div>
        ''')
        
        # Analysis section
        f.write('''
            <div class="section">
                <h2>Statistical Analysis</h2>
        ''')
        
        # Calculate stats if we have enough data
        valid_scales = [scale for scale in results['time_scales'] if scale is not None]
        if valid_scales:
            mean_scale = np.mean(valid_scales)
            std_scale = np.std(valid_scales)
            min_scale = np.min(valid_scales)
            max_scale = np.max(valid_scales)
            
            f.write(f'''
                <p>Mean time scale: <span class="metric">{mean_scale:.4f}</span></p>
                <p>Standard deviation: <span class="metric">{std_scale:.4f}</span></p>
                <p>Minimum value: <span class="metric">{min_scale:.4f}</span></p>
                <p>Maximum value: <span class="metric">{max_scale:.4f}</span></p>
                <p>Coefficient of variation: <span class="metric">{(std_scale/mean_scale*100):.2f}%</span></p>
            ''')
        else:
            f.write("<p>Insufficient data for statistical analysis</p>")
        
        f.write('''
            </div>
        ''')
        
        # Visualizations section
        f.write('''
            <div class="section">
                <h2>Visualizations</h2>
                <div class="container">
        ''')
        
        # Add visualizations
        for img_file in glob.glob(f"{viz_dir}/*.png"):
            img_name = os.path.basename(img_file)
            rel_path = os.path.relpath(img_file, OUTPUT_ROOT)
            f.write(f'''
                    <div class="image-container">
                        <h3>{img_name.replace('_', ' ').replace('.png', '')}</h3>
                        <img src="../{rel_path}" alt="{img_name}">
                    </div>
            ''')
        
        f.write('''
                </div>
            </div>
        ''')
        
        # Links to individual experiment results
        f.write('''
            <div class="section">
                <h2>Individual Experiment Results</h2>
                <ul>
        ''')
        
        for i, output_dir in enumerate(results['output_dirs']):
            time_pair = results['time_points'][i]
            report_file = glob.glob(f"{output_dir}/reports/*_summary_report.html")
            if report_file:
                rel_path = os.path.relpath(report_file[0], OUTPUT_ROOT)
                f.write(f'''
                        <li><a href="../{rel_path}">Detailed results for {time_pair}</a></li>
                ''')
        
        f.write('''
                </ul>
            </div>
        ''')
        
        # Interpretation section
        f.write('''
            <div class="section">
                <h2>Interpretation</h2>
        ''')
        
        if valid_scales:
            cov = std_scale/mean_scale*100
            if cov < 10:
                f.write(f'''
                    <p>The time scale parameter appears to be <strong>very stable</strong> across different time point pairs for this patient, 
                    with a coefficient of variation of only {cov:.2f}%.</p>
                    <p>This suggests that the relationship between model time and physical time is consistent 
                    throughout the progression of the disease for this patient.</p>
                ''')
            elif cov < 25:
                f.write(f'''
                    <p>The time scale parameter shows <strong>moderate stability</strong> across different time point pairs for this patient, 
                    with a coefficient of variation of {cov:.2f}%.</p>
                    <p>Some variation is present, but the overall relationship between model time and physical time 
                    appears to be relatively consistent during disease progression.</p>
                ''')
            else:
                f.write(f'''
                    <p>The time scale parameter shows <strong>significant variation</strong> across different time point pairs for this patient, 
                    with a coefficient of variation of {cov:.2f}%.</p>
                    <p>This suggests that the relationship between model time and physical time may be changing 
                    during the progression of the disease, possibly indicating changes in tumor growth dynamics.</p>
                ''')
        else:
            f.write("<p>Insufficient data to provide a meaningful interpretation.</p>")
        
        f.write('''
            </div>
        ''')
        
        # Close HTML
        f.write('''
        </body>
        </html>
        ''')
    
    return html_path

def main():
    """Main function to run the experiment and analysis"""
    print(f"Starting time scale stability experiment for {PATIENT_ID}")
    print(f"Running {len(TIME_POINT_PAIRS)} time point pairs: {TIME_POINT_PAIRS}")
    
    # Run the experiment
    results = run_experiment()
    
    # Create visualizations
    viz_dir = visualize_time_scales(results)
    
    # Create the summary report
    report_path = create_summary_report(results, viz_dir)
    
    print("\nExperiment completed!")
    print(f"Summary report: {report_path}")
    
    # Print time scale comparison
    print("\nTime Scale Comparison:")
    for i, (time_pair, time_scale) in enumerate(zip(results['time_points'], results['time_scales'])):
        print(f"  {time_pair}: {time_scale:.6f if time_scale is not None else 'N/A'}")
    
    # If we have valid results, print statistics
    valid_scales = [scale for scale in results['time_scales'] if scale is not None]
    if valid_scales:
        mean_scale = np.mean(valid_scales)
        std_scale = np.std(valid_scales)
        cv = std_scale/mean_scale*100
        print(f"\nMean time scale: {mean_scale:.6f}")
        print(f"Standard deviation: {std_scale:.6f}")
        print(f"Coefficient of variation: {cv:.2f}%")

if __name__ == "__main__":
    main()