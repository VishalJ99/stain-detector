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
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return OmegaConf.create(config_dict)


def update_config_with_args(config, args):
    """Update configuration with command line arguments if provided"""
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.optimizer.lr = args.lr
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.device:
        config.training.device = args.device
    if args.data_dir:
        config.data.train_dir = os.path.join(args.data_dir, "train")
        config.data.val_dir = os.path.join(args.data_dir, "val")
        config.data.test_dir = os.path.join(args.data_dir, "test")
    if args.overfit_single_batch:
        config.training.overfit_single_batch = True

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
