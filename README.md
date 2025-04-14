# StainDetector

A deep learning system for detecting and classifying histological stains in whole slide images.

## Project Structure

```
├── configs/                  # Configuration files
│   └── train_config.yaml     # Config for training & inference defaults
├── data/                     # Root directory for patch data (tracked by DVC)
│   ├── train/                # Training patches (Subdirs: PAS/, H&E/, IHC/, etc.)
        ├── PAS/
        ├── ...
│   ├── val/                  # Validation patches (same structure as train)
│   └── test/                 # Test patches (same structure as train)
├── example_wsis/             # Example directory for WSI inference input
├── notebooks/                # Jupyter notebooks for exploration
├── src/                      # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration handling
│   ├── dataset.py            # PyTorch Dataset class for loading patches
│   ├── model.py              # Model definitions
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Main evaluation script (on patches)
│   ├── inference.py          # Inference script (on WSIs)
│   └── utils.py              # Helper functions (repro checks, WSI tiling utils)
├── runs/                     # Output directory for all runs
```
Update this
## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd StainDetector
```

2. Create a conda environment with all dependencies:
```bash
conda env create -f environment.yml
conda activate stain-detector
```

## Usage

### Training

Train a model using the default configuration:

```bash
python src/train.py --config configs/train_config.yaml
```

Override configuration parameters:

```bash
python src/train.py --config configs/train_config.yaml --run_name my_custom_run --batch_size 32 --lr 0.0005
```

#### Sanity Check

To perform a quick sanity check by overfitting to a single batch:

```bash
python src/train.py --overfit_single_batch
```

This will train the model on a single batch repeatedly, using the same batch for both training and validation. The model should quickly achieve near-perfect accuracy on this batch, verifying that the training pipeline is working correctly.

### Evaluation

Evaluate a trained model on test data:

```bash
python src/evaluate.py --checkpoint runs/my_run_name/checkpoints/best_model.pth --config configs/train_config.yaml
```

### Inference on Whole Slide Images

Run inference on a single WSI:

```bash
python src/inference.py --checkpoint runs/my_run_name/checkpoints/best_model.pth --wsi_path path/to/slide.svs
```

Process all WSIs in a directory:

```bash
python src/inference.py --checkpoint runs/my_run_name/checkpoints/best_model.pth --wsi_path path/to/wsi_directory
```

Save extracted patches:

```bash
python src/inference.py --checkpoint runs/my_run_name/checkpoints/best_model.pth --wsi_path path/to/slide.svs --save_patches
```

## Configuration

The project uses a central YAML configuration file in `configs/train_config.yaml`. Key parameters include:

- **Data parameters**: Paths, image size, batch size
- **Model parameters**: Architecture, pretrained weights
- **Training parameters**: Optimizer, learning rate, scheduler settings
- **Inference parameters**: Patch size, tissue thresholds, WSI settings

See the config file for details on all available parameters.

## Environment Variables and API Keys

This project uses environment variables for managing API keys and sensitive configuration. We use `python-dotenv` for loading these variables from a `.env` file. Create your own `.env` file based on the provided template:

```bash
# Copy the template file
cp .env.template .env

# Edit the .env file to add your API keys
nano .env  # or use any text editor
```

The following environment variables are supported:

- `WANDB_API_KEY`: Weights & Biases API key for experiment tracking
- `WANDB_ENTITY`: Your Weights & Biases username or organization
- `WANDB_PROJECT`: (Optional) Override the W&B project name

The `.env` file is included in `.gitignore` to prevent accidental commits of sensitive information.
