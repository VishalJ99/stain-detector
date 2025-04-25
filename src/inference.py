import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

from config import get_base_parser, load_config_from_args
from model import StainClassifier
from utils import (
    check_git_dvc_clean,
    create_reproduce_command,
    extract_patches_from_wsi,
    get_device,
    log_unclean_state,
    save_git_dvc_state,
    set_seed,
    setup_logging,
)


def predict_patch(model, patch, transform, device):
    """
    Predict class for a single patch

    Args:
        model (nn.Module): Trained model
        patch (numpy.ndarray): Image patch (RGB format)
        transform (callable): Transform to apply to the patch
        device (torch.device): Device to use for inference

    Returns:
        tuple: (predicted_class_idx, prediction_probabilities)
    """
    # Preprocess the patch
    patch_tensor = transform(patch).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(patch_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    return pred_class, probs.cpu().numpy()[0]


def process_wsi(
    wsi_path,
    model,
    config,
    transform,
    device,
    output_dir,
    class_names,
    logger=None,
    save_patches=False,
):
    """
    Process a whole slide image for inference

    Args:
        wsi_path (str): Path to the WSI file
        model (nn.Module): Trained model
        config: Configuration object
        transform (callable): Transform to apply to patches
        device (torch.device): Device to use for inference
        output_dir (str): Directory to save results
        class_names (list): List of class names

    Returns:
        pandas.DataFrame: DataFrame with patch predictions
    """
    # Extract patches from WSI
    patches = extract_patches_from_wsi(
        wsi_path=wsi_path,
        patch_size=config.inference_defaults.patch_size,
        level=config.inference_defaults.level,
        overlap=config.inference_defaults.overlap,
        num_patches=config.inference_defaults.num_patches,
        tissue_threshold=config.inference_defaults.tissue_threshold,
        create_debug_images=save_patches,
        debug_output_dir=os.path.join(output_dir, "debug", Path(wsi_path).stem)
        if save_patches
        else None,
        save_patches_dir=os.path.join(output_dir, "patches", Path(wsi_path).stem)
        if save_patches
        else None,
    )

    # Prepare results storage
    results = []
    slide_name = Path(wsi_path).stem

    # Process each patch
    for i, patch in enumerate(tqdm(patches, desc=f"Processing {slide_name}")):
        # Get prediction
        pred_class, pred_probs = predict_patch(model, patch, transform, device)

        # Save result
        result = {
            "slide_name": slide_name,
            "patch_idx": i,
            "predicted_class": class_names[pred_class],
            "predicted_class_idx": pred_class,
        }

        # Add probabilities for each class
        for j, class_name in enumerate(class_names):
            result[f"prob_{class_name}"] = pred_probs[j]

        results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    if len(results) > 0:
        csv_path = os.path.join(output_dir, f"{slide_name}_results.csv")
        results_df.to_csv(csv_path, index=False)

    return results_df


def main():
    """Main inference function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Stain Classification WSI Inference", parents=[get_base_parser()]
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--wsi_path",
        type=str,
        required=True,
        help="Path to the WSI file or directory containing WSIs",
    )

    parser.add_argument(
        "--save_patches",
        action="store_true",
        help="Save patches to disk",
    )

    args = parser.parse_args()

    if "example_wsis" not in args.wsi_path:
        # Running on an actual case, so check for uncommitted changes.
        is_clean, details = check_git_dvc_clean()
        if not is_clean:
            log_unclean_state(details)
            sys.exit(1)

    save_patches = args.save_patches

    # Load configuration
    config = load_config_from_args(args)

    # Set up output directory
    run_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(run_dir, "inferences", timestamp)

    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    logger = setup_logging(
        name="inference", log_dir=output_dir, log_file_name="inference.log"
    )
    logger.info("Starting WSI inference")
    logger.info(f"Using checkpoint: {args.checkpoint}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint["classes"]

    # Set random seed
    set_seed(config.training.seed)

    # Record environment state
    save_git_dvc_state(output_dir)

    # Create reproducibility file
    create_reproduce_command(args, os.path.join(output_dir, "reproduce_inference.txt"))

    # Get device
    device = get_device(config.training.device)
    logger.info(f"Using device: {device}")

    # TODO: Move this normalisation to a transforms.py script and avoid hardcoding.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create model
    logger.info(f"Creating model: {config.model.name}")
    model = StainClassifier(config)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Process WSIs
    all_results = []

    # Check if input is a directory or single file
    if os.path.isdir(args.wsi_path):
        # Use glob to find all WSI files in the directory and subdirectories.
        wsi_files = glob.glob(
            os.path.join(args.wsi_path, "**", "*.svs"), recursive=True
        )
        wsi_files += glob.glob(
            os.path.join(args.wsi_path, "**", "*.tif"), recursive=True
        )
        logger.info(f"Found {len(wsi_files)} WSI files in {args.wsi_path}")

        # Process all WSI files in directory
        wsi_files = [
            os.path.join(args.wsi_path, f)
            for f in os.listdir(args.wsi_path)
            if f.lower().endswith((".svs", ".tif", ".tiff", ".ndpi"))
        ]
        logger.info(f"Found {len(wsi_files)} WSI files in {args.wsi_path}")

    else:
        wsi_files = [args.wsi_path]
        logger.info(f"Processing WSI: {args.wsi_path}")

    # Process each WSI file.
    for wsi_file in wsi_files:
        logger.info(f"Processing WSI: {wsi_file}")
        results_df = process_wsi(
            wsi_file,
            model,
            config,
            transform,
            device,
            output_dir,
            class_names,
            logger,
            save_patches,
        )
        all_results.append(results_df)

    # Combine all results
    if all_results:
        # Save patch-level results
        combined_results = pd.concat(all_results, ignore_index=True)
        patch_csv_path = os.path.join(output_dir, "all_results_patch_level.csv")
        combined_results.to_csv(patch_csv_path, index=False)

        # Generate WSI-level results (mode of patch predictions)
        wsi_results = []
        for slide_name in combined_results["slide_name"].unique():
            slide_patches = combined_results[
                combined_results["slide_name"] == slide_name
            ]
            # Get most common predicted class
            mode_class_idx = slide_patches["predicted_class_idx"].mode().iloc[0]
            mode_class = class_names[mode_class_idx]

            wsi_result = {
                "slide_name": slide_name,
                "predicted_class": mode_class,
                "predicted_class_idx": mode_class_idx,
                "patch_count": len(slide_patches),
            }
            wsi_results.append(wsi_result)

        wsi_df = pd.DataFrame(wsi_results)
        wsi_csv_path = os.path.join(output_dir, "all_results_wsi.csv")
        wsi_df.to_csv(wsi_csv_path, index=False)

        logger.info(f"Patch-level results saved to: {patch_csv_path}")
        logger.info(f"WSI-level results saved to: {wsi_csv_path}")

    logger.info(f"Inference completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
