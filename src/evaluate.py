import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from datetime import datetime
from omegaconf import OmegaConf
import json

from dataset import StainDataset
from model import StainClassifier
from utils import (
    set_seed, 
    get_device, 
    setup_logging, 
    save_git_dvc_state,
    create_reproduce_command,
    save_metrics,
    log_batch_images
)
from config import load_config

def evaluate(model, data_loader, criterion, device, logger):
    """
    Evaluate model on dataset
    
    Args:
        model (nn.Module): Model to evaluate
        data_loader (DataLoader): Data loader
        criterion: Loss function
        device (torch.device): Device to evaluate on
        logger (logging.Logger): Logger
        
    Returns:
        tuple: (loss, accuracy, all_predictions, all_targets)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()
            
            # Save predictions and targets
            all_preds.append(torch.softmax(outputs, dim=1).cpu())
            all_targets.append(targets.cpu())
            
            # Log first batch images
            if batch_idx == 0:
                log_batch_images(inputs, targets, predicted, data_loader.dataset.classes, 0, prefix="test")
            
            # Log progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(data_loader):
                logger.info(f"Batch {batch_idx+1}/{len(data_loader)}, "
                            f"Loss: {loss.item():.4f}, "
                            f"Acc: {100.*correct/total:.2f}%")
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    logger.info(f"Evaluation completed - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return loss, accuracy, all_preds, all_targets

def main():
    """Main evaluation function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stain Classification Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--test_dir', type=str,
                        help='Directory with test data (overrides config)')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Override config with command line arguments
    if args.test_dir:
        config.data.test_dir = args.test_dir
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        run_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(run_dir, "evaluations", timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting evaluation")
    logger.info(f"Using checkpoint: {args.checkpoint}")
    
    # Set random seed
    set_seed(config.training.seed)
    
    # Record environment state
    save_git_dvc_state(output_dir)
    
    # Create reproducibility file
    create_reproduce_command(args, os.path.join(output_dir, "reproduce_eval.txt"))
    
    # Get device
    device = get_device(config.training.device)
    logger.info(f"Using device: {device}")
    
    # Create test dataset and data loader
    test_dataset = StainDataset(
        data_dir=config.data.test_dir,
        image_size=config.data.image_size,
        mode="test"
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    logger.info(f"Creating model: {config.model.name}")
    model = StainClassifier(config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    logger.info("Starting evaluation...")
    loss, accuracy, all_preds, all_targets = evaluate(model, test_loader, criterion, device, logger)
    
    # Save metrics and visualizations
    logger.info("Saving metrics and visualizations...")
    save_metrics(all_preds, all_targets, test_dataset.classes, output_dir)
    
    # Save basic metrics to a separate file
    basic_metrics = {
        "accuracy": float(accuracy / 100),  # Convert from percentage
        "loss": float(loss),
        "test_samples": len(test_dataset),
        "checkpoint": args.checkpoint,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
        json.dump(basic_metrics, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to: {output_dir}")
    
    return loss, accuracy

if __name__ == "__main__":
    main()