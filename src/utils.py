import json
import logging
import os
import random
import subprocess
import sys
import warnings
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import seaborn as sns
import torch
import torch.optim as optim
from PIL import Image, ImageDraw
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

import wandb


def set_seed(seed):
    """
    Set random seeds for reproducibility

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device=None):
    """
    Get the device to use for training/inference

    Args:
        device (str, optional): Device specified in config

    Returns:
        torch.device: Device to use
    """
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def setup_logging(name, log_dir=None, log_file_name=None):
    """
    Set up root logging configuration

    Args:
        name (str): Name of the logger (typically __name__)
        log_dir (str, optional): Directory to save log files
        log_file_name (str, optional): Name of the log file

    Returns:
        logging.Logger: Configured logger instance
    """
    # Configure the logger with the provided name
    logger = logging.getLogger(name)

    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)

    # Create a consistent log format for both console and file
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        if not log_file_name:
            log_file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        log_file = os.path.join(log_dir, log_file_name)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def check_git_dvc_clean():
    """
    Check if git and dvc have uncommitted changes.

    Returns:
        tuple: (is_clean, details_dict)
            - is_clean (bool): True if both git and dvc are clean
            - details_dict (dict): Details about git and dvc state
    """
    details = {}
    is_clean = True

    # Check git status
    try:
        git_status = (
            subprocess.check_output(["git", "status", "--porcelain"])
            .decode("utf-8")
            .strip()
        )

        git_is_clean = not git_status
        details["git"] = {
            "is_clean": git_is_clean,
            "status": git_status.split("\n") if git_status else [],
        }

        if not git_is_clean:
            is_clean = False

    except (subprocess.CalledProcessError, FileNotFoundError):
        details["git"] = {"error": "Git information not available"}

    # Check dvc status
    try:
        dvc_status = subprocess.check_output(["dvc", "status"]).decode("utf-8").strip()
        dvc_is_clean = not dvc_status or "up to date" in dvc_status.lower()

        details["dvc"] = {
            "is_clean": dvc_is_clean,
            "status": dvc_status.split("\n") if dvc_status else ["Up to date"],
        }

        if not dvc_is_clean:
            is_clean = False

    except (subprocess.CalledProcessError, FileNotFoundError):
        details["dvc"] = {"error": "DVC information not available"}

    return is_clean, details


def get_dvc_file(dataset_path):
    """Derive DVC file path from dataset path."""
    # Strip trailing slashes for consistent handling
    clean_path = dataset_path.rstrip("/")

    # Add .dvc extension to get the DVC file path
    dvc_file = f"{clean_path}.dvc"

    # Verify the DVC file exists
    if not os.path.exists(dvc_file):
        warnings.warn(f"DVC file not found for dataset: {dvc_file}")

    return dvc_file


def log_unclean_state(details):
    """
    Log warnings about uncommitted changes in git/dvc repositories.

    Args:
        details (dict): Details about git/dvc state from check_git_dvc_clean
    """
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("git-dvc-check")

    logger.warning("⚠️  Uncommitted changes detected in git/dvc repositories!")
    logger.warning("-" * 80)
    if details["git"].get("status"):
        logger.warning("Git changes:")
        for item in details["git"]["status"]:
            logger.warning(f"  {item}")

    logger.warning("")
    if details["dvc"].get("status") and "Up to date" not in details["dvc"]["status"]:
        logger.warning("DVC changes:")
        for item in details["dvc"]["status"]:
            if item:  # Skip empty lines
                logger.warning(f"  {item}")

    # Add a blank line before the final warning
    logger.warning("-" * 80)

    logger.warning(
        "⚠️  To ensure reproducibility, please commit all changes "
        "before running or use a debug flag for experimentation."
    )


def save_git_dvc_state(output_dir):
    """
    Save git and dvc state information for reproducibility

    Args:
        output_dir (str): Directory to save state information
    """
    state = {}

    # Get git information
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        git_branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        git_status = (
            subprocess.check_output(["git", "status", "--porcelain"])
            .decode("utf-8")
            .strip()
        )

        state["git"] = {
            "hash": git_hash,
            "branch": git_branch,
            "status": git_status.split("\n") if git_status else [],
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        state["git"] = {"error": "Git information not available"}

    # Get dvc information
    try:
        dvc_status = subprocess.check_output(["dvc", "status"]).decode("utf-8").strip()
        state["dvc"] = {
            "status": dvc_status.split("\n") if dvc_status else "Up to date"
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        state["dvc"] = {"error": "DVC information not available"}

    # Add timestamp
    state["timestamp"] = datetime.now().isoformat()

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "git_dvc_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def create_reproduce_command(args, output_file):
    """
    Create a text file with the command to reproduce this run

    Args:
        args (argparse.Namespace): Command line arguments
        output_file (str): File to save reproduction command
    """
    command = ["python"]

    # Get script name
    script_path = sys.argv[0]
    command.append(script_path)

    # Add all args
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            command.append(f"--{arg_name.replace('_', '-')} {arg_value}")

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(" ".join(command))


def get_optimizer(model, config):
    """
    Get optimizer based on configuration

    Args:
        model (torch.nn.Module): Model to optimize
        config: Configuration object with optimizer parameters

    Returns:
        torch.optim.Optimizer: Optimizer
    """
    # Default to Adam optimizer
    lr = config.optimizer.lr
    weight_decay = config.optimizer.weight_decay
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, config):
    """
    Get learning rate scheduler

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        config: Configuration object

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler or None
    """
    # Return CosineAnnealingLR if defined in config
    if hasattr(config, "scheduler") and config.scheduler is not None:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.scheduler.params.T_max
        )
    return None


def log_batch_images(inputs, targets, predictions, class_names, epoch, prefix="val"):
    """
    Log a batch of images to wandb

    Args:
        inputs (torch.Tensor): Input images
        targets (torch.Tensor): Target labels
        predictions (torch.Tensor): Predicted labels
        class_names (list): List of class names
        epoch (int): Current epoch
        prefix (str): Prefix for wandb logging (val or train)
    """
    # Convert tensors to numpy arrays
    images = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # Create a list of wandb.Image objects with captions
    wandb_images = []
    for i in range(min(8, len(images))):  # Log up to 8 images
        img = np.transpose(images[i], (1, 2, 0))  # CHW to HWC
        # Denormalize the image (assuming ImageNet normalization)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        target_name = class_names[targets[i]]
        pred_name = class_names[predictions[i]]
        caption = f"True: {target_name}, Pred: {pred_name}"

        wandb_images.append(wandb.Image(img, caption=caption))

    wandb.log({f"{prefix}/sample_images": wandb_images, f"{prefix}/epoch": epoch})


def is_tissue(patch, threshold=0.05):
    """
    Determine if a patch contains tissue based on HSV color thresholding

    Args:
        patch (numpy.ndarray): RGB image patch
        threshold (float): Minimum tissue percentage threshold

    Returns:
        bool: True if the patch contains enough tissue, False otherwise
    """
    # Convert to HSV
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)

    # Create a mask for tissue (exclude background)
    # Background is typically very bright and low saturation
    mask = (hsv[:, :, 0] < 30) & (hsv[:, :, 1] > 30) & (hsv[:, :, 2] < 220)

    # Calculate tissue percentage
    tissue_percentage = np.sum(mask) / mask.size

    return tissue_percentage > threshold


def extract_patches_from_wsi(
    wsi_path,
    patch_size=256,
    overlap=0.25,
    level=0,
    tissue_threshold=0.05,
    create_debug_images=True,
    debug_output_dir=None,
    num_patches=1000,
    save_patches_dir=None,
    logger=None,
):
    """
    Extract random patches from tissue regions in a whole slide image (WSI)

    Args:
        wsi_path (str): Path to the WSI file
        patch_size (int): Size of the patches to extract
        overlap (float): Overlap between patches (0-1)
        level (int): WSI pyramid level to extract from
        tissue_threshold (float): Minimum tissue percentage threshold
        create_debug_images (bool): Whether to create debug overlay images
        debug_output_dir (str, optional): Directory to save debug images
        num_patches (int): Maximum number of patches to extract
        save_patches_dir (str, optional): Directory to save extracted patches as images

    Returns:
        list: List of tissue patches as numpy arrays
    """
    # Create output directory if needed for debug images
    if create_debug_images:
        assert (
            debug_output_dir is not None
        ), "debug_output_dir must be provided when create_debug_images=True"
        os.makedirs(debug_output_dir, exist_ok=True)

    # Create directory for saving patches if specified
    if save_patches_dir:
        os.makedirs(save_patches_dir, exist_ok=True)

    # Open the slide
    if logger:
        logger.info(f"Opening slide: {wsi_path}")
    slide = openslide.OpenSlide(wsi_path)
    width, height = slide.level_dimensions[level]

    # Initialize tracking
    patches = []
    count = 0

    # Calculate downsample factor based on slide size
    scale_factor = 1 / 16 if create_debug_images else 1 / min(32, width // 4000)
    thumb_width = int(width * scale_factor)
    thumb_height = int(height * scale_factor)

    if logger:
        logger.info(f"Creating thumbnail at resolution {thumb_width}x{thumb_height}")
    thumbnail = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
    thumbnail_np = np.array(thumbnail)

    # Create debug overlay image
    if create_debug_images:
        downsampled = thumbnail.copy()
        draw = ImageDraw.Draw(downsampled)

    # Apply tissue detection to thumbnail
    hsv = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    tissue_mask = (saturation > 20) & (value < 230)

    # Optional: Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    # Save the tissue mask if in debug mode
    if create_debug_images:
        mask_path = os.path.join(debug_output_dir, "tissue_mask_thumbnail.png")
        Image.fromarray((tissue_mask * 255).astype(np.uint8)).save(mask_path)
        thumbnail.save(os.path.join(debug_output_dir, "thumbnail.png"))

    # Find coordinates of all tissue pixels in the thumbnail
    tissue_coords = np.where(tissue_mask)
    tissue_points = list(zip(tissue_coords[1], tissue_coords[0]))  # (x, y) format

    if not tissue_points:
        if logger:
            logger.info("No tissue regions found in the slide")
        slide.close()
        return patches

    # Calculate thumbnail patch size for checking neighboring pixels
    thumb_patch_size = int(patch_size * scale_factor)

    # Randomly sample from tissue regions
    if logger:
        logger.info(
            f"Randomly sampling up to {num_patches} patches from tissue regions"
        )

    # Keep track of already sampled regions to avoid overlap
    sampled_regions = set()
    max_attempts = num_patches * 10  # Limit attempts to avoid infinite loop
    attempts = 0

    while count < num_patches and attempts < max_attempts:
        attempts += 1

        # Randomly select a tissue point from the mask
        if not tissue_points:
            break

        point_idx = np.random.randint(0, len(tissue_points))
        thumb_x, thumb_y = tissue_points[point_idx]

        # Check if we have enough space for a patch
        if (
            thumb_x + thumb_patch_size >= thumbnail_np.shape[1]
            or thumb_y + thumb_patch_size >= thumbnail_np.shape[0]
        ):
            continue

        # Verify this region has enough tissue
        region = tissue_mask[
            thumb_y : thumb_y + thumb_patch_size, thumb_x : thumb_x + thumb_patch_size
        ]
        tissue_percentage = np.sum(region) / region.size

        if tissue_percentage <= tissue_threshold:
            continue

        # Map to full resolution coordinates
        full_x = int(thumb_x / scale_factor)
        full_y = int(thumb_y / scale_factor)

        # Create a region key to avoid overlap
        region_key = (full_x // (patch_size // 4), full_y // (patch_size // 4))
        if region_key in sampled_regions:
            continue

        sampled_regions.add(region_key)

        # Extract full resolution patch
        patch_pil = slide.read_region(
            (full_x, full_y), level, (patch_size, patch_size)
        ).convert("RGB")
        patch_np = np.array(patch_pil)

        # Final verification on the full resolution patch
        should_infer = is_tissue_patch(patch_np, tissue_threshold)

        if should_infer:
            patches.append(patch_np)

            # Save patch to disk if directory is provided
            if save_patches_dir:
                patch_filename = f"patch_{count}_x{full_x}_y{full_y}.png"
                patch_path = os.path.join(save_patches_dir, patch_filename)
                Image.fromarray(patch_np).save(patch_path)

            count += 1

        # Draw debug visualization
        if create_debug_images:
            rect = [
                thumb_x,
                thumb_y,
                thumb_x + thumb_patch_size,
                thumb_y + thumb_patch_size,
            ]
            draw.rectangle(rect, outline="green" if should_infer else "red", width=1)

    # Save debug overlay
    if create_debug_images:
        debug_path = os.path.join(debug_output_dir, "patch_debug_overlay.png")
        downsampled.save(debug_path)

    if logger:
        logger.info(
            f"Extracted {count} patches from {wsi_path} after {attempts} attempts"
        )
    slide.close()
    return patches


def is_tissue_patch(patch_np, threshold=0.05):
    """
    Determine if a patch contains tissue based on HSV thresholding

    Args:
        patch_np (numpy.ndarray): RGB image patch
        threshold (float): Minimum tissue percentage threshold

    Returns:
        bool: True if the patch contains enough tissue, False otherwise
    """
    # HSV-based tissue filtering (using original criteria)
    hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    tissue_mask = (saturation > 20) & (value < 230)
    tissue_percentage = np.sum(tissue_mask) / (patch_np.shape[0] * patch_np.shape[1])
    return tissue_percentage > threshold


def save_metrics(all_preds, all_targets, class_names, output_dir):
    """
    Calculate and save evaluation metrics

    Args:
        all_preds (torch.Tensor): Model predictions (after softmax)
        all_targets (torch.Tensor): Ground truth labels
        class_names (list): List of class names
        output_dir (str): Directory to save metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensors to numpy
    if isinstance(all_preds, torch.Tensor):
        all_preds = all_preds.cpu().numpy()
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()

    # Get class predictions
    pred_classes = np.argmax(all_preds, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, pred_classes)

    # Compute accuracy
    accuracy = np.sum(pred_classes == all_targets) / len(all_targets)

    # Compute per-class metrics
    metrics = {"accuracy": float(accuracy), "per_class": {}}

    for i, class_name in enumerate(class_names):
        # One-vs-rest approach for multi-class metrics
        class_preds = all_preds[:, i]
        class_targets = (all_targets == i).astype(int)

        # ROC and AUC
        fpr, tpr, _ = roc_curve(class_targets, class_preds)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(class_targets, class_preds)
        ap = average_precision_score(class_targets, class_preds)

        # True positives, false positives, etc.
        tp = np.sum((pred_classes == i) & (all_targets == i))
        fp = np.sum((pred_classes == i) & (all_targets != i))
        tn = np.sum((pred_classes != i) & (all_targets != i))
        fn = np.sum((pred_classes != i) & (all_targets == i))

        # Sensitivity, specificity, etc.
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (
            2 * (precision_val * sensitivity) / (precision_val + sensitivity)
            if (precision_val + sensitivity) > 0
            else 0
        )

        metrics["per_class"][class_name] = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision_val),
            "f1": float(f1),
            "auc": float(roc_auc),
            "average_precision": float(ap),
        }

    # Save metrics to JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Plot and save ROC curves
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        class_targets = (all_targets == i).astype(int)
        fpr, tpr, _ = roc_curve(class_targets, all_preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()

    # Plot and save PR curves
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        class_targets = (all_targets == i).astype(int)
        precision, recall, _ = precision_recall_curve(class_targets, all_preds[:, i])
        ap = average_precision_score(class_targets, all_preds[:, i])
        plt.plot(recall, precision, lw=2, label=f"{class_name} (AP = {ap:.2f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"))
    plt.close()


def get_run_name_from_checkpoint(checkpoint_path):
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    run_name = os.path.basename(run_dir).split("_", 1)[-1]
    return run_name
