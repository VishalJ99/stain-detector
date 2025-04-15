import argparse
import os

import yaml
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Load environment variables from .env file
load_dotenv()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Stain Detection Training and Inference"
    )

    # Add cli arguments for common config changes.
    # All additions here must be added to the load_config function.
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration YAML file",
    )
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
        "--overfit_single_batch",
        action="store_true",
        help="Overfit to a single batch as a sanity check",
    )
    parser.add_argument(
        "--save_patches",
        action="store_true",
        help="Save patches to disk",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        help="Number of patches to sample per WSI",
    )

    return parser.parse_args()


def load_config(args):
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

    return config


def update_config_with_args(config, args):
    """TODO: Fix, v messy and error prone way of handling args."""
    """Update configuration with command line arguments if provided"""
    if hasattr(args, "batch_size"):
        if args.batch_size is not None:
            config.data.batch_size = args.batch_size
    if hasattr(args, "lr"):
        if args.lr is not None:
            config.optimizer.lr = args.lr
    if hasattr(args, "epochs"):
        if args.epochs is not None:
            config.training.num_epochs = args.epochs
    if hasattr(args, "device"):
        if args.device is not None:
            config.training.device = args.device
    if hasattr(args, "data_dir"):
        if args.data_dir is not None:
            print(f"Overriding data directory to: {args.data_dir}")
            config.data.data_dir = args.data_dir
    if hasattr(args, "overfit_single_batch"):
        if args.overfit_single_batch:
            config.training.overfit_single_batch = args.overfit_single_batch
    if hasattr(args, "num_patches"):
        if args.num_patches is not None:
            config.inference_defaults.num_patches = args.num_patches
    if hasattr(args, "save_patches"):
        if args.save_patches:
            config.inference_defaults.save_patches = args.save_patches

    # Add environment variable overrides for sensitive data
    apply_env_overrides(config)

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
    # For example:
    # if os.environ.get('AWS_ACCESS_KEY_ID'):
    #     config.aws.access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')

    return config
