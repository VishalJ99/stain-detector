import argparse
import os

import yaml
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Load environment variables from .env file
load_dotenv()


def get_base_parser():
    """
    Create a base argument parser with common config override options.
    This can be used by all scripts to ensure consistent CLI options.

    Returns:
        argparse.ArgumentParser: Base parser with common arguments
    """
    parser = argparse.ArgumentParser(add_help=False)

    # Common config overrides
    parser.add_argument("--run_dir", type=str, help="Override run directory in config")
    parser.add_argument("--batch_size", type=int, help="Override batch size in config")
    parser.add_argument("--lr", type=float, help="Override learning rate in config")
    parser.add_argument(
        "--epochs", type=int, help="Override number of epochs in config"
    )
    parser.add_argument("--device", type=str, help="Override device in config")
    parser.add_argument(
        "--data_dir", type=str, help="Override data directory in config"
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        help="Number of patches to sample per WSI",
    )

    return parser


def load_config_from_args(args):
    """Load configuration from YAML file"""
    # Check if args has config attribute before accessing it
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    config = OmegaConf.create(config_dict)

    # Add environment variable overrides for sensitive data
    apply_env_overrides(config)

    # Update config with command line arguments if provided
    config = update_config_with_args(config, args)

    # Expand config with derived paths
    config = expand_config(config)

    return config


def update_config_with_args(config, args):
    """Update configuration with command line arguments if provided"""
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.optimizer.lr = args.lr
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.device is not None:
        config.training.device = args.device
    if args.data_dir is not None:
        print(f"Overriding data directory to: {args.data_dir}")
        config.data.data_dir = args.data_dir
    if args.num_patches is not None:
        config.inference_defaults.num_patches = args.num_patches
    if args.run_dir is not None:
        config.logging.run_dir = args.run_dir

    return config


def expand_config(config):
    """Expand configuration with derived paths"""
    # Ensure data.data_dir exists in config
    if not hasattr(config, "data") or not hasattr(config.data, "data_dir"):
        raise ValueError("Configuration must include data.data_dir")

    # Create derived directory paths based on data_dir
    data_dir = config.data.data_dir

    # Add standard dataset subdirectories
    config.data.train_dir = os.path.join(data_dir, "train")
    config.data.val_dir = os.path.join(data_dir, "val")
    config.data.test_dir = os.path.join(data_dir, "test")
    return config


def apply_env_overrides(config):
    """Apply environment variable overrides to config"""
    # Weights & Biases settings
    if os.environ.get("WANDB_API_KEY"):
        # Just having the environment variable set is enough for wandb
        pass

    if os.environ.get("WANDB_PROJECT"):
        config.logging.wandb_project = os.environ.get("WANDB_PROJECT")

    if os.environ.get("WANDB_ENTITY"):
        config.logging.wandb_entity = os.environ.get("WANDB_ENTITY")

    # Additional API keys or secrets can be added here
    return config
