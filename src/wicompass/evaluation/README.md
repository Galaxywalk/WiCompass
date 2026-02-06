# VQ-VAE Model Evaluation Package

Unified VQ-VAE model evaluation, encoding, and visualization package with clean, efficient interfaces.

## File Structure

```
src/wicompass/evaluation/
├── README.md                  # This documentation
├── __init__.py               # Package exports
├── core.py                   # Core functionality module
├── evaluation_visualizer.py  # Evaluation result visualization
├── pose_visualization.py     # Pose visualization module
├── token_visualization.py    # Token distribution visualization
├── evaluate_model.py         # Model evaluation script
├── encode_dataset.py         # Dataset encoding script
└── visualize_poses.py        # Pose visualization script (with --simple mode)
└── visualize_tokens.py       # Token visualization script
```

## Quick Start

### Command-Line Interface

```bash
# Model evaluation
python -m wicompass.evaluation.evaluate_model \
    --model work_dirs/best_model.pth \
    --config configs/joint_vae_base.json

# Dataset encoding
python -m wicompass.evaluation.encode_dataset \
    --model work_dirs/best_model.pth \
    --config configs/joint_vae_base.json \
    --output encoded_tokens.h5

# Pose visualization (simple mode, no model required)
python -m wicompass.evaluation.visualize_poses \
    --config configs/joint_vae_base.json \
    --simple

# Pose visualization (with reconstruction comparison)
python -m wicompass.evaluation.visualize_poses \
    --config configs/joint_vae_base.json \
    --model work_dirs/best_model.pth

# Token distribution visualization
python -m wicompass.evaluation.visualize_tokens \
    --model work_dirs/best_model.pth \
    --config configs/joint_vae_base.json
```

## Python API

### One-Click Evaluation and Encoding

```python
from wicompass.evaluation import evaluate_model, encode_dataset

# Model evaluation
results = evaluate_model(
    model_path="work_dirs/best_model.pth",
    config_path="configs/joint_vae_base.json",
    test_ratio=0.1
)

# Dataset encoding
stats = encode_dataset(
    model_path="work_dirs/best_model.pth", 
    config_path="configs/joint_vae_base.json",
    output_path="tokens.h5"
)
```

### Custom Evaluation

```python
from wicompass.evaluation import (
    load_config, load_model, ModelEvaluator, 
    create_evaluation_dataset, create_dataloader
)
from wicompass.model import JointVQVAELoss

# Load model
config = load_config("configs/joint_vae_base.json")
model = load_model(config['model'], "work_dirs/best_model.pth")
criterion = JointVQVAELoss()

# Create dataset and dataloader
dataset, dataset_info, enabled_datasets = create_evaluation_dataset(
    "configs/joint_vae_base.json"
)
dataloader = create_dataloader(dataset, batch_size=64)

# Evaluate
evaluator = ModelEvaluator(model, criterion)
results = evaluator.evaluate_dataset(dataloader)
```

### Custom Encoding

```python
from wicompass.evaluation import (
    load_config, load_model, ModelEncoder,
    create_evaluation_dataset, create_dataloader, save_tokens
)

# Load model and dataset
config = load_config("configs/joint_vae_base.json")
model = load_model(config['model'], "work_dirs/best_model.pth")
dataset, _, _ = create_evaluation_dataset("configs/joint_vae_base.json")
dataloader = create_dataloader(dataset, batch_size=128)

# Encode
encoder = ModelEncoder(model)
results = encoder.encode_dataset(dataloader)

# Save
save_tokens(results['tokens'], results['labels'], "tokens.h5")
```

### Pose Visualization

```python
from wicompass.evaluation import PoseVisualizer

# Create visualizer (with model for reconstruction comparison)
visualizer = PoseVisualizer(
    config_path="configs/joint_vae_base.json",
    model_path="work_dirs/best_model.pth"  # Optional
)

# Load and visualize samples
samples = visualizer.load_dataset_samples(num_samples=6)
visualizer.visualize_sample_poses(samples, "pose_output/")
```

### Token Distribution Visualization

```python
from wicompass.evaluation import TokenVisualizer

# Create token visualizer
visualizer = TokenVisualizer(
    config_path="configs/joint_vae_base.json",
    model_path="work_dirs/best_model.pth"
)

# Create t-SNE visualization
df = visualizer.create_token_visualization(
    output_dir="token_viz/",
    method='tsne',
    sampling_mode='per-dataset',
    samples_per_dataset=1000
)
```

## Core Modules

### core.py - Core Module

#### Configuration and Data
- `load_config()` - Load config file (JSON/YAML)
- `create_evaluation_dataset()` - Create evaluation dataset
- `create_dataloader()` - Create data loader
- `split_dataset()` - Split dataset into train/test

#### Model Operations
- `load_model()` - Load model from checkpoint
- `ModelEvaluator` - Model evaluation class
- `ModelEncoder` - Model encoding class
- `BaseVisualizer` - Base class for visualization tools

#### Data I/O
- `save_tokens()` - Save tokens to HDF5
- `load_tokens()` - Load tokens from HDF5

#### Analysis Functions
- `analyze_joint_errors()` - Joint-level error analysis
- `analyze_sample_errors()` - Sample-level error analysis

#### High-Level Functions
- `evaluate_model()` - One-click model evaluation
- `encode_dataset()` - One-click dataset encoding

### evaluation_visualizer.py - Evaluation Visualization

- `plot_token_heatmap()` - Token-Codebook usage heatmap
- `plot_joint_errors()` - Joint error distribution plot
- `plot_sample_errors()` - Sample error analysis plot
- `create_evaluation_report()` - Generate complete evaluation report

### pose_visualization.py - Pose Visualization

- `plot_single_pose()` - Draw a single 3D pose
- `plot_pose_comparison()` - Compare original and reconstructed poses
- `plot_multiple_poses()` - Draw multiple poses overview
- `PoseVisualizer` - Human pose visualizer class

### token_visualization.py - Token Visualization

- `extract_token_representations()` - Extract token features
- `apply_dimensionality_reduction()` - Apply t-SNE/PCA
- `create_token_distribution_visualization()` - Create distribution plots
- `TokenVisualizer` - Token distribution visualizer class

## Output Files

### Evaluation Results
- `evaluation_report.json` - Detailed evaluation report
- `token_heatmap.png` - Token usage heatmap
- `joint_errors.png` - Joint error distribution
- `sample_errors.png` - Sample error analysis
- `*.npy` - Raw data files

### Encoding Results
- `*.h5` - Encoded tokens (HDF5 format with metadata)

## Constants

```python
from wicompass.evaluation import JOINT_NAMES, BONE_CONNECTIONS

# 22 joint names
JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', ...]

# Bone connections (parent, child)
BONE_CONNECTIONS = [(0, 1), (0, 2), (0, 3), ...]
```

## Notes

1. **Dependencies**: Requires PyTorch, matplotlib, h5py, numpy
2. **Optional**: scikit-learn (for t-SNE/PCA), plotly (for interactive plots)
3. **Config Format**: Supports JSON and YAML
4. **Device Support**: Auto-detects CUDA availability
5. **Memory**: Adjust batch_size for large datasets

## Usage Tips

- **Daily Use**: Use command-line scripts or high-level functions
- **Custom Development**: Use classes from core.py
- **Batch Processing**: Use ModelEvaluator and ModelEncoder classes
- **Result Analysis**: Use visualization modules for reports
