# Temporal Calibration of Biophysical Tumor Growth Models via Inverse Fitting


> **A computational framework for modeling brain tumor growth dynamics through inverse fitting of the Fisher-Kolmogorov (FK) equation on longitudinal MRI scans with flexible temporal weighting schemes.**

[**Documentation**](#usage) | [**Visualization Examples**](#visualization)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training Mode](#training-mode)
  - [Forward Prediction](#forward-prediction)
  - [Train-Test Evaluation](#train-test-evaluation)
  - [Adjacent Weeks Experiment](#adjacent-weeks-experiment)
- [Data Format](#data-format)
- [Model Architecture](#model-architecture)
- [Loss Functions](#loss-functions)
- [Temporal Weighting Strategies](#temporal-weighting-strategies)
- [Visualization](#visualization)
- [Advanced Configuration](#advanced-configuration)

---

## Overview

This repository implements a sophisticated inverse problem solver for calibrating reaction-diffusion tumor growth models to serial brain MRI observations. The framework addresses a critical challenge in computational oncology: **mapping between model time and real clinical time** while accounting for patient-specific tumor dynamics.

### The Problem

Traditional biophysical models operate in dimensionless "model time" units, making direct comparison with clinical observations challenging. This framework introduces temporal calibration parameters that:

- **Estimate tumor initiation time** (`t_start`) relative to the first observation
- **Scale model time to real time** (`time_scale`) to account for individual growth dynamics
- **Weight multiple time points flexibly** to prioritize critical observations (e.g., recent scans for prediction)

### The Solution

We solve an inverse problem using **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** to optimize:

1. **Spatial parameters**: Tumor seed location (NxT1_pct, NyT1_pct, NzT1_pct)
2. **Biophysical parameters**: Diffusion coefficient (Dw), proliferation rate (ρ)
3. **Temporal parameters**: Start time (t_start), time scale factor (time_scale)

The framework supports multiple loss functions (Dice, Soft Dice, Hausdorff, boundary-weighted) and flexible temporal weighting strategies to accommodate various clinical scenarios.


---

## Key Features

**Flexible Temporal Calibration**
- Map between model time and real clinical time
- Estimate tumor initiation time before first observation
- Support for multiple temporal weighting strategies

**Multiple Loss Functions**
- Standard Dice coefficient
- Soft Dice (robust to minor registration errors)
- Boundary-weighted loss (reduced penalty near tumor margins)
- Hausdorff distance (shape similarity)
- Combined loss (weighted ensemble)

**Efficient Optimization**
- CMA-ES evolutionary algorithm
- Parallel evaluation with configurable workers
- Early stopping with patience mechanism

**Comprehensive Evaluation**
- Train-test split validation
- Adjacent time point experiments
- Rich visualization suite
- Detailed performance metrics (Dice, Hausdorff, volume error)

**Clinical Applicability**
- Future time point prediction
- Patient-specific parameter estimation

---

## Quick Start

### Minimal Example

```python
from TimeSeriesFittingSolver import TimeSeriesFittingSolver
import nibabel as nib

# Load your data
wm = nib.load('path/to/white_matter.nii.gz').get_fdata()
gm = nib.load('path/to/gray_matter.nii.gz').get_fdata()
tumor_week0 = nib.load('path/to/tumor_week0.nii.gz').get_fdata()
tumor_week5 = nib.load('path/to/tumor_week5.nii.gz').get_fdata()

# Configure settings
settings = {
    "rho0": 0.06,
    "dw0": 1.0,
    "sigma0": 0.3,
    "generations": 25,
    "workers": 9,
    "loss_type": "soft_dice",
    "resolution_factor": {0: 0.3, 0.4: 0.5, 0.7: 0.7}
}

# Initialize solver
time_points = [0, 5]  # weeks
tumor_data = [tumor_week0, tumor_week5]
solver = TimeSeriesFittingSolver(settings, wm, gm, tumor_data, time_points)

# Run optimization
predicted_series, results = solver.run()

# Results contain optimized parameters and loss values
print(f"Optimal Dw: {results['opt_params'][3]:.6f}")
print(f"Optimal rho: {results['opt_params'][4]:.6f}")
print(f"Minimum loss: {results['minLoss']:.4f}")
```

---

## Usage

### Training Mode

Train the model on longitudinal MRI data to estimate patient-specific parameters:

```bash
python TimeSeriesFitting.py \
    --data_dir /path/to/data \
    --patient_id Patient-091 \
    --output_dir ./results/Patient-091 \
    --time_points 0 2 5 7 \
    --loss_type combined \
    --generations 25 \
    --workers 9 \
    --weighting_strategy inverse \
    --target_time 7
```

**Parameters:**
- `--data_dir`: Base directory containing patient data
- `--patient_id`: Patient identifier
- `--output_dir`: Directory to save results
- `--time_points`: Time points (weeks) to use for training
- `--loss_type`: Loss function (`dice`, `soft_dice`, `boundary`, `hausdorff`, `combined`)
- `--generations`: Number of CMA-ES generations
- `--workers`: Number of parallel workers
- `--weighting_strategy`: Time point weighting (`equal`, `inverse`, `exponential`, `log`)
- `--target_time`: Target time for inverse distance weighting (default: max time point)

**[Insert Figure: Training convergence plot showing loss over generations]**

### Forward Prediction

Use trained parameters to predict tumor growth at multiple time points:

```bash
python forward.py \
    --data_dir /path/to/data \
    --patient_id Patient-091 \
    --output_dir ./predictions \
    --time_points 0 2 5 7 10 14 \
    --param_file ./results/Patient-091/best_params_results.npy \
    --predict_week 21
```

**Parameters:**
- `--param_file`: Path to trained parameters (.npy file)
- `--predict_week`: Future time point to predict (optional)
- `--slice_index`: Slice for visualization (default: auto-select)

**Output:** Predicted tumor masks as NIfTI files and visualization plots.

**[Insert Figure: Forward prediction showing actual vs predicted tumor masks]**

### Train-Test Evaluation

Evaluate model generalization with train-test splits:

```bash
python train_test_eval.py \
    --data_dir /path/to/data \
    --patient_id Patient-078 \
    --output_dir ./evaluation \
    --train_weeks 0 1 5 19 29 37 \
    --test_weeks 41 45 56 62 \
    --generations 25 \
    --workers 9 \
    --loss_type combined \
    --weighting_strategy inverse
```

**Features:**
- Trains on specified weeks
- Tests on held-out time points
- Generates comprehensive performance metrics
- Creates train-test comparison visualizations

**Output Metrics:**
- Dice coefficient (spatial overlap)
- Hausdorff distance (shape similarity)
- Volume error (percentage)
- Train/test performance ratio

**[Insert Figure: Train-test performance comparison plot]**

### Adjacent Weeks Experiment

Systematically evaluate model extrapolation capability:

```bash
python adjacent_weeks_experiment.py \
    --data_dir /path/to/data \
    --patient_id Patient-091 \
    --output_dir ./experiments \
    --weeks 0 2 5 7 10 14 \
    --generations 15 \
    --workers 4
```

**What it does:**
- Trains on adjacent pairs: (0,2)→5, (2,5)→7, (5,7)→10, etc.
- Evaluates extrapolation performance
- Creates summary visualizations across experiments
- Analyzes relationship between extrapolation distance and accuracy

**[Insert Figure: Extrapolation distance vs test loss scatter plot]**

---

## Data Format

### Directory Structure

The framework expects the following directory structure:

```
data_dir/
└── Patient-XXX/
    ├── week-000/
    │   ├── seg_mask.nii.gz      # Tumor segmentation mask
    │   ├── T1_pve_1.nii.gz      # Gray matter map
    │   └── T1_pve_2.nii.gz      # White matter  map
    ├── week-002/
    │   ├── seg_mask.nii.gz
    │   ├── T1_pve_1.nii.gz
    │   └── T1_pve_2.nii.gz
    └── week-XXX/
        └── ...
```

### File Specifications

- **seg_mask.nii.gz**: Binary or probabilistic tumor mask (0-1 range)
- **T1_pve_1.nii.gz**: Gray matter tissue probability (0-1 range)
- **T1_pve_2.nii.gz**: White matter tissue probability (0-1 range)
- All files must be co-registered and in the same image space
- Recommended resolution: 1mm³ isotropic

### Preprocessing Requirements

1. **Skull stripping**: Brain extraction completed
2. **Tissue segmentation**: FSL FAST or equivalent
3. **Registration**: All time points aligned to common space
4. **Tumor segmentation**: Manual or semi-automated delineation

---

## Model Architecture

### Fisher-Kolmogorov (FK) Equation

The tumor growth is modeled using the reaction-diffusion FK equation:

$$
\frac{\partial u}{\partial t} = \nabla \cdot (D(\mathbf{x}) \nabla u) + \rho u(1 - u)
$$

Where:
- $u(\mathbf{x}, t)$: Tumor cell density at position $\mathbf{x}$ and time $t$
- $D(\mathbf{x})$: Spatially-varying diffusion coefficient
- $\rho$: Proliferation rate
- $\nabla$: Gradient operator

### Tissue-Dependent Diffusion

$$
D(\mathbf{x}) = D_w \cdot \text{WM}(\mathbf{x}) + D_g \cdot \text{GM}(\mathbf{x})
$$

Where $D_w$ and $D_g$ are diffusion coefficients in white and gray matter, typically with $D_w = 10 \times D_g$.

### Temporal Calibration

The model introduces temporal parameters to map model time $t_m$ to real clinical time $t_c$:

$$
t_m = (t_c - t_{\text{start}}) \times s_{\text{time}}
$$

Where:
- $t_{\text{start}}$: Tumor initiation time relative to first observation
- $s_{\text{time}}$: Time scale factor

### Numerical Solver

- **Method**: Finite Difference with explicit time-stepping
- **Resolution**: Multi-scale progressive refinement (0.3 → 0.5 → 0.7)
- **Boundary Conditions**: Zero-flux (Neumann)
- **Initialization**: Gaussian seed at estimated origin

**[Insert Figure: Schematic of FK equation solver with tissue maps]**

---

## Loss Functions

### 1. Standard Dice Coefficient

$$
\text{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B|}
$$

**Loss:** $L_{\text{dice}} = 1 - \text{Dice}(A, B)$

### 2. Soft Dice (Registration-Robust)

Dilates both masks by margin $m$ to tolerate small misregistration:

$$
\text{SoftDice}(A, B) = \alpha \cdot \text{Dice}(A, B) + (1-\alpha) \cdot \max(\text{Dice}(A \oplus m, B), \text{Dice}(A, B \oplus m))
$$

### 3. Boundary-Weighted Loss

Reduces penalty near tumor boundaries using distance transforms:

$$
L_{\text{boundary}} = \frac{\sum_{\mathbf{x}} |A(\mathbf{x}) - B(\mathbf{x})| \cdot w(\mathbf{x})}{\sum_{\mathbf{x}} w(\mathbf{x})}
$$

Where $w(\mathbf{x})$ decreases near boundaries.

### 4. Hausdorff Distance

Modified Hausdorff using 95th percentile for robustness:

$$
H_{95}(A, B) = \max(h_{95}(A, B), h_{95}(B, A))
$$

### 5. Combined Loss

Weighted ensemble for comprehensive evaluation:

$$
L_{\text{combined}} = 0.4 L_{\text{dice}} + 0.3 L_{\text{soft}} + 0.2 L_{\text{boundary}} + 0.1 L_{\text{Hausdorff}}
$$

**[Insert Figure: Visual comparison of different loss functions on example data]**

---

## Temporal Weighting Strategies

### 1. Equal Weighting

All time points contribute equally:

$$
w_i = \frac{1}{N} \quad \forall i
$$

### 2. Inverse Distance Weighting

Prioritizes time points closer to target time $t_{\text{target}}$:

$$
w_i = \frac{1}{|t_{\text{target}} - t_i|^p + \epsilon}
$$

Normalized: $\tilde{w}_i = w_i / \sum_j w_j$

### 3. Exponential Recency

Exponentially higher weights for recent observations:

$$
w_i = e^{\beta (t_i - t_{\min})}
$$

### 4. Logarithmic Distance

Balanced importance using log-scale distances:

$$
w_i = \frac{1}{\log(|t_{\text{target}} - t_i| + e)}
$$

**Use Cases:**
- **Equal**: Retrospective fitting with no specific target
- **Inverse**: Prediction at specific future time point
- **Exponential**: Emphasize recent progression for short-term forecasting
- **Logarithmic**: Balanced attention across all time points



---

## Visualization

The framework generates comprehensive visualizations automatically in output_dir/visualizations:

### 1. Optimization Convergence

**[Insert Figure: convergence.png]**

Tracks best and mean fitness across generations, showing optimization progress.

### 2. Time Point Loss Analysis

**[Insert Figure: timepoint_losses.png]**

Individual loss values at each observed time point, revealing which observations are well-fit.

### 3. Parameter Evolution

**[Insert Figure: parameter_evolution.png]**

Trajectories of mean parameter values during optimization.

### 4. Spatial Error Maps

**[Insert Figure: spatial_error_map.png]**

Color-coded visualization:
- **Green**: True positive (correctly predicted tumor)
- **Red**: False positive (over-prediction)
- **Blue**: False negative (missed tumor)

### 5. Volume Trajectory

**[Insert Figure: volume_trajectory.png]**

Comparison of predicted vs actual tumor volumes over time.

### 6. Train-Test Performance

**[Insert Figure: train_test_performance.png]**

Side-by-side comparison of model performance on training and testing weeks.

---

## Advanced Configuration

### Multi-Resolution Strategy

Progressive refinement for computational efficiency:

```python
settings["resolution_factor"] = {
    0.0: 0.3,    # Generations 0-40%: Low resolution
    0.4: 0.5,    # Generations 40-70%: Medium resolution
    0.7: 0.7     # Generations 70-100%: High resolution
}
```

### Early Stopping

Prevent overfitting with patience-based stopping:

```python
settings["early_stopping"] = True
settings["patience"] = 5  # Stop if no improvement for 5 generations
```

### Custom Loss Weights

Adjust combined loss function weights:

```python
settings["dice_weight"] = 0.5
settings["soft_weight"] = 0.2
settings["boundary_weight"] = 0.2
settings["hausdorff_weight"] = 0.1
```

### Parameter Ranges

Constrain optimization search space:

```python
settings["parameterRanges"] = [
    [0, 1],              # NxT1_pct
    [0, 1],              # NyT1_pct
    [0, 1],              # NzT1_pct
    [0.001, 3],          # Dw
    [0.0001, 0.225],     # rho
    [-100, 0],           # t_start
    [0.1, 10]            # time_scale
]
```

### CMA-ES Tuning

```python
settings["sigma0"] = 0.3      # Initial step size
settings["workers"] = 9       # Parallel evaluations
settings["generations"] = 25  # Maximum iterations
```

---

## Acknowledgments

- Fisher-Kolmogorov equation solver adapted from [TumorGrowthToolkit](TumorGrowthToolkit/)

---

