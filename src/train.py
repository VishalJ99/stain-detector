import argparse
import os
import random
import sys
import warnings
from datetime import datetime

import petname
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

import wandb
from config import get_base_parser, load_config_from_args
from dataset import StainDataset
from model import StainClassifier
from utils import (
    check_git_dvc_clean,
    create_reproduce_command,
    get_device,
    get_optimizer,
    get_scheduler,
    log_unclean_state,
    save_git_dvc_state,
    set_seed,
    setup_logging,
)

warnings.filterwarnings("ignore", category=UserWarning)


def train_epoch(
    model, train_loader, criterion, optimizer, device, config, epoch, logger
):
    """
    Train the model for one epoch

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        config: Configuration object
        epoch (int): Current epoch
        logger (logging.Logger): Logger

    Returns:
        float: Average training loss for this epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % config.logging.log_interval_steps == 0:
            logger.info(
                f"Epoch {epoch+1}/{config.training.num_epochs}, "
                f"Batch {batch_idx+1}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}, "
                f"Acc: {100.*correct/total:.2f}%"
            )

            # Log metrics to wandb
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/acc": 100.0 * correct / total,
                    "train/step": epoch * len(train_loader) + batch_idx,
                }
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    # Log epoch metrics
    wandb.log(
        {
            "train/epoch_loss": avg_loss,
            "train/epoch_acc": accuracy,
            "train/epoch": epoch,
        }
    )

    return avg_loss


def validate(model, val_loader, criterion, device, config, epoch, logger):
    """
    Validate the model

    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        device (torch.device): Device to validate on
        config: Configuration object
        epoch (int): Current epoch
        logger (logging.Logger): Logger

    Returns:
        tuple: (val_loss, val_accuracy)
    """
    # Access classes attribute safely for confusion matrix.
    # Handle the case where we're using a Subset (for overfit_single_batch).
    if hasattr(val_loader.dataset, "classes"):
        class_names = val_loader.dataset.classes
    elif hasattr(val_loader.dataset, "dataset") and hasattr(
        val_loader.dataset.dataset, "classes"
    ):
        # For Subset objects that reference the original dataset.
        class_names = val_loader.dataset.dataset.classes
    else:
        # Fallback to numeric class names.
        num_classes = config.data.num_classes
        class_names = [str(i) for i in range(num_classes)]
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Save predictions and targets for metrics.
            all_preds.append(torch.softmax(outputs, dim=1).cpu())
            all_targets.append(targets.cpu())

            # Log sample images periodically.
            if batch_idx == 0 and epoch % 5 == 0:
                from utils import log_batch_images

                log_batch_images(
                    inputs,
                    targets,
                    predicted,
                    class_names,
                    epoch,
                    prefix="val",
                )

    # Concatenate predictions and targets.
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    val_loss = total_loss / len(val_loader)
    val_accuracy = 100.0 * correct / total

    # Log validation metrics.
    wandb.log(
        {
            "val/loss": val_loss,
            "val/acc": val_accuracy,
            "val/epoch": epoch,
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                probs=all_preds.numpy(),
                y_true=all_targets.numpy(),
                class_names=class_names,
            ),
        }
    )

    logger.info(
        f"Validation - Epoch {epoch+1}/{config.training.num_epochs}, "
        f"Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%"
    )

    return val_loss, val_accuracy


def train(config, args):
    """
    Main training function

    Args:
        config: Configuration object
        args: Command line arguments
    """

    # Fetch timestamp for the run.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = petname.generate(2, separator="-")  # e.g., "elegant-turtle"
    run_dir = os.path.join(config.logging.run_dir, f"{timestamp}_{run_name}")
    os.makedirs(run_dir)

    # Initialize W&B experiment.
    wandb_run = wandb.init(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True),
        dir=run_dir,
        name=run_name,
    )

    # Add the run name to the config.
    config.logging.wandb_run_name = run_name

    logger = setup_logging(name="train", log_dir=run_dir, log_file_name="train.log")
    logger.info(f"Starting training run: {run_name}")
    logger.info(f"W&B URL: {wandb_run.url}")

    # Set random seed for reproducibility.
    set_seed(config.training.seed)

    # Save git and dvc status.
    save_git_dvc_state(run_dir)

    # Save the exact config used for this run.
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config))

    # Create a file with the command to reproduce this run.
    create_reproduce_command(args, os.path.join(run_dir, "reproduce_train.txt"))

    # Get device.
    device = get_device(config.training.device)
    logger.info(f"Using device: {device}")

    # Create datasets and data loaders.
    logger.info("Creating datasets and data loaders...")

    # Check if num_classes is already defined in config
    expected_num_classes = config.data.get("num_classes", None)

    train_dataset = StainDataset(
        data_dir=config.data.train_dir,
        image_size=config.data.image_size,
        mode="train",
        expected_num_classes=expected_num_classes,
    )

    val_dataset = StainDataset(
        data_dir=config.data.val_dir,
        image_size=config.data.image_size,
        mode="val",
        expected_num_classes=expected_num_classes,
    )
    assert len(train_dataset.classes) == len(val_dataset.classes), (
        f"Train and val sets have different number of classes: "
        f"{len(train_dataset.classes)} vs {len(val_dataset.classes)}"
    )

    if args.overfit_single_batch:
        logger.info("OVERFIT MODE: Training on a single batch for sanity check")
        # Create a DataLoader with a subset of the train dataset (just 1 batch)
        # Use random indices to get a diverse batch of samples
        all_indices = list(range(len(train_dataset)))
        random.shuffle(all_indices)
        indices = all_indices[: min(config.data.batch_size, len(train_dataset))]

        single_batch_dataset = torch.utils.data.Subset(train_dataset, indices)

        train_loader = torch.utils.data.DataLoader(
            single_batch_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,  # No need to shuffle as we're using the same batch
            num_workers=4,
            pin_memory=True,
        )

        # Use the same batch for validation
        val_loader = train_loader
        logger.warning(
            "Using the same batch for training and validation in overfit mode"
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    # Create model.
    logger.info(f"Creating model: {config.model.name}")
    model = StainClassifier(config)
    model = model.to(device)

    # Log model parameters and gradients to W&B for visualization.
    wandb.watch(model, log="all", log_freq=config.logging.log_interval_steps)

    # Create loss function, optimizer, and scheduler.
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Create directory for checkpoints.
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Training loop.
    logger.info("Starting training...")
    best_val_accuracy = 0

    for epoch in range(config.training.num_epochs):
        # Train for one epoch.
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch, logger
        )

        # Validate.
        val_loss, val_accuracy = validate(
            model, val_loader, criterion, device, config, epoch, logger
        )

        # Update learning rate.
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log learning rate.
        wandb.log({"train/lr": optimizer.param_groups[0]["lr"], "train/epoch": epoch})

        # Save checkpoint.
        checkpoint_path = os.path.join(
            checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "config": OmegaConf.to_container(config, resolve=True),
                "classes": train_dataset.classes,
            },
            checkpoint_path,
        )

        # Save best model.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accuracy,
                    "config": OmegaConf.to_container(config, resolve=True),
                    "classes": train_dataset.classes,
                },
                best_model_path,
            )
            logger.info(
                f"Saved best model with validation accuracy: {val_accuracy:.2f}%"
            )

    logger.info(
        f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%"
    )
    wandb.finish()


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Train a stain classifier", parents=[get_base_parser()]
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file",
    )

    parser.add_argument(
        "--overfit_single_batch",
        action="store_true",
        help="Overfit to a single batch as a sanity check",
    )

    args = parser.parse_args()

    # Check for uncommitted changes if we're not in debug mode
    if not args.overfit_single_batch:
        is_clean, details = check_git_dvc_clean()
        if not is_clean:
            log_unclean_state(details)
            sys.exit(1)

    # Load configuration from file.
    config = load_config_from_args(args)

    # Start training.
    train(config, args)
