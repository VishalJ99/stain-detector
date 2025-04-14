import os
import json
import random
import logging
import numpy as np
import torch
import torch.optim as optim
import cv2
import openslide
from datetime import datetime
import wandb
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

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

def setup_logging(log_dir=None):
    """
    Set up logging configuration
    
    Args:
        log_dir (str, optional): Directory to save log files
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger("stain_detector")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"stain_detector_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_git_dvc_state(output_dir):
    """
    Save git and dvc state information for reproducibility
    
    Args:
        output_dir (str): Directory to save state information
    """
    state = {}
    
    # Get git information
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
        git_status = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
        
        state["git"] = {
            "hash": git_hash,
            "branch": git_branch,
            "status": git_status.split("\n") if git_status else []
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
            optimizer, 
            T_max=config.scheduler.params.T_max
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

def extract_patches_from_wsi(wsi_path, patch_size, mpp=None, level=0, overlap=0, num_patches=1000, 
                            tissue_threshold=0.05, save_dir=None):
    """
    Extract patches from a whole slide image (WSI)
    
    Args:
        wsi_path (str): Path to the WSI file
        patch_size (int): Size of the patches to extract
        mpp (float, optional): Target microns-per-pixel (for resolution normalization)
        level (int): WSI pyramid level to extract from
        overlap (float): Overlap between patches (0-1)
        num_patches (int): Maximum number of patches to extract
        tissue_threshold (float): Minimum tissue percentage threshold
        save_dir (str, optional): Directory to save patches if needed
        
    Returns:
        list: List of patch images as numpy arrays
    """
    # Open the WSI
    slide = openslide.OpenSlide(wsi_path)
    
    # Handle resolution normalization if mpp is specified
    if mpp is not None and hasattr(slide, 'properties'):
        base_mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, mpp))
        base_mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, mpp))
        
        # Calculate the level that most closely matches the desired mpp
        base_mpp = (base_mpp_x + base_mpp_y) / 2
        scale_factor = mpp / base_mpp
        
        # Find closest level
        level_dimensions = slide.level_dimensions
        level_downsamples = slide.level_downsamples
        
        for i in range(len(level_downsamples)):
            if level_downsamples[i] >= scale_factor:
                level = i
                break
    
    # Get dimensions at the target level
    width, height = slide.level_dimensions[level]
    
    # Calculate step size with overlap consideration
    step_size = int(patch_size * (1 - overlap))
    
    # Generate grid coordinates
    x_coords = list(range(0, width - patch_size + 1, step_size))
    y_coords = list(range(0, height - patch_size + 1, step_size))
    
    # Shuffle coordinates for random sampling
    coords = [(x, y) for x in x_coords for y in y_coords]
    random.shuffle(coords)
    
    # Extract patches
    patches = []
    slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
    
    for i, (x, y) in enumerate(coords):
        if len(patches) >= num_patches:
            break
            
        # Extract patch
        patch = np.array(slide.read_region((x * int(slide.level_downsamples[level]), 
                                            y * int(slide.level_downsamples[level])), 
                                            level, (patch_size, patch_size)))
        
        # Convert from RGBA to RGB
        patch = patch[:, :, :3]
        
        # Check if patch contains enough tissue
        if is_tissue(patch, threshold=tissue_threshold):
            patches.append(patch)
            
            # Save patch if save_dir is provided
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                patch_filename = f"{slide_name}_level{level}_x{x}_y{y}.png"
                patch_path = os.path.join(save_dir, patch_filename)
                cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    
    slide.close()
    return patches

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
    metrics = {
        "accuracy": float(accuracy),
        "per_class": {}
    }
    
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
        f1 = 2 * (precision_val * sensitivity) / (precision_val + sensitivity) if (precision_val + sensitivity) > 0 else 0
        
        metrics["per_class"][class_name] = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision_val),
            "f1": float(f1),
            "auc": float(roc_auc),
            "average_precision": float(ap)
        }
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
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