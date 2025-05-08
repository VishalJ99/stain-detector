#!/usr/bin/env python3
# patch_extractor.py - Extract patches from whole slide images
import sys
import json
import os
import argparse
import cv2
import numpy as np
import openslide
from PIL import Image, ImageDraw
from tqdm import tqdm

MASK_SAT = 0
MASK_VAL = 245


def is_tissue_patch(patch_np, threshold=0.15):
    hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    tissue_mask = (saturation > MASK_SAT) & (value < MASK_VAL)
    tissue_percentage = np.sum(tissue_mask) / (patch_np.shape[0] * patch_np.shape[1])
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
    exclusion_conditions=None,
    exclusion_mode="any",
    extraction_mode="random",
    save_patches=False,
    output_dir=None,
    label=None,
):
    """
    Extract patches from tissue regions in a whole slide image (WSI)

    Args:
        wsi_path (str): Path to the WSI file
        patch_size (int): Size of the patches to extract
        overlap (float): Overlap between patches (0-1)
        level (int): WSI pyramid level to extract from
        tissue_threshold (float): Minimum tissue percentage threshold
        create_debug_images (bool): Whether to create debug overlay images
        debug_output_dir (str, optional): Directory to save debug images
        num_patches (int): Maximum number of patches to extract (only used in random mode)
        exclusion_conditions (list): List of tuples (coord, operator, value) for exclusion criteria
                                  e.g. [('x', '<', 33500)] to exclude patches with x < 33500
                                  Coordinates are at base/original resolution (level 0)
        exclusion_mode (str): 'any' to exclude if any condition is met, 'all' for all conditions
        extraction_mode (str): 'random' to extract random patches, 'contiguous' for grid-based patches
        save_patches (bool): Whether to save patches to disk
        output_dir (str, optional): Directory to save patches and metadata
        label (str, optional): Optional label/class for the patches (used for organizing output)

    Returns:
        If save_patches=False:
            list: List of tuples (patch_np, x, y) where patch_np is the numpy array and x,y are coordinates
        If save_patches=True:
            tuple: (patch_list, metadata_dict)
    """
    # Initialize exclusion conditions if not provided
    if exclusion_conditions is None:
        exclusion_conditions = []

    # Validate exclusion_mode
    if exclusion_mode not in ["any", "all"]:
        print(
            f"Warning: Invalid exclusion_mode '{exclusion_mode}', defaulting to 'any'"
        )
        exclusion_mode = "any"

    # Validate extraction_mode
    if extraction_mode not in ["random", "contiguous"]:
        print(
            f"Warning: Invalid extraction_mode '{extraction_mode}', defaulting to 'random'"
        )
        extraction_mode = "random"

    # Setup for saving patches
    metadata = {}
    if save_patches:
        assert (
            output_dir is not None
        ), "output_dir must be provided when save_patches=True"
        os.makedirs(output_dir, exist_ok=True)

        # Create slide-specific subdirectory using slide name
        slide_name = os.path.splitext(os.path.basename(wsi_path))[0]

        # If label is provided, organize by label
        if label is not None:
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            slide_output_dir = os.path.join(label_dir, slide_name)
        else:
            slide_output_dir = os.path.join(output_dir, slide_name)

        os.makedirs(slide_output_dir, exist_ok=True)

        # Initialize metadata
        metadata = {
            "slide_path": wsi_path,
            "slide_name": slide_name,
            "label": label,
            "patch_size": patch_size,
            "level": level,
            "overlap": overlap,
            "extraction_mode": extraction_mode,
            "patches": [],
        }

    # Create output directory if needed for debug images
    if create_debug_images:
        assert (
            debug_output_dir is not None
        ), "debug_output_dir must be provided when create_debug_images=True"
        os.makedirs(debug_output_dir, exist_ok=True)

    # Open the slide
    print(f"Opening slide: {wsi_path}")
    slide = openslide.OpenSlide(wsi_path)
    width, height = slide.level_dimensions[level]

    # Initialize tracking
    patches = []
    count = 0

    # Calculate downsample factor based on slide size
    scale_factor = 1 / 16 if create_debug_images else 1 / min(32, width // 4000)
    thumb_width = int(width * scale_factor)
    thumb_height = int(height * scale_factor)

    print(f"Creating thumbnail at resolution {thumb_width}x{thumb_height}")
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
    tissue_mask = (saturation > MASK_SAT) & (value < MASK_VAL)

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
        print("No tissue regions found in the slide")
        slide.close()
        if save_patches:
            return patches, metadata
        return patches

    # Calculate thumbnail patch size for checking neighboring pixels
    thumb_patch_size = int(patch_size * scale_factor)

    # Different extraction strategies based on mode
    if extraction_mode == "random":
        print(f"Randomly sampling up to {num_patches} patches from tissue regions")

        # Keep track of already sampled regions to avoid overlap
        sampled_regions = set()
        max_attempts = num_patches * 100  # Limit attempts to avoid infinite loop
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
                thumb_y : thumb_y + thumb_patch_size,
                thumb_x : thumb_x + thumb_patch_size,
            ]
            tissue_percentage = np.sum(region) / region.size

            if tissue_percentage <= tissue_threshold:
                continue

            # Map to full resolution coordinates
            full_x = int(thumb_x / scale_factor)
            full_y = int(thumb_y / scale_factor)

            # Check exclusion conditions on base-level coordinates
            should_exclude = False

            # Track conditions that are satisfied
            satisfied_conditions = []

            for condition in exclusion_conditions:
                coord, operator, value = condition

                # Get the appropriate coordinate value
                if coord.lower() == "x":
                    coord_value = full_x
                elif coord.lower() == "y":
                    coord_value = full_y
                else:
                    continue

                # Apply the operator comparison
                condition_met = False
                if operator == "<" and coord_value < value:
                    condition_met = True
                elif operator == ">" and coord_value > value:
                    condition_met = True
                elif operator == "<=" and coord_value <= value:
                    condition_met = True
                elif operator == ">=" and coord_value >= value:
                    condition_met = True
                elif operator == "==" and coord_value == value:
                    condition_met = True

                if condition_met:
                    satisfied_conditions.append(True)
                else:
                    satisfied_conditions.append(False)

            # Determine exclusion based on mode
            if exclusion_mode == "any" and any(satisfied_conditions):
                should_exclude = True
            elif (
                exclusion_mode == "all"
                and all(satisfied_conditions)
                and satisfied_conditions
            ):
                should_exclude = True

            if should_exclude:
                if create_debug_images and len(satisfied_conditions) > 0:
                    # Draw excluded regions in blue
                    rect = [
                        thumb_x,
                        thumb_y,
                        thumb_x + thumb_patch_size,
                        thumb_y + thumb_patch_size,
                    ]
                    draw.rectangle(rect, outline="blue", width=1)
                continue

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
                # Store patch with coordinates
                patches.append((patch_np, full_x, full_y))
                print(
                    f"Patch {count} at position x={full_x}, y={full_y} is a tissue patch"
                )

                # Save patch if requested
                if save_patches:
                    patch_filename = f"{slide_name}_x{full_x}_y{full_y}_l{level}.png"
                    patch_path = os.path.join(slide_output_dir, patch_filename)
                    patch_pil.save(patch_path)

                    # Store metadata
                    patch_info = {
                        "filename": patch_filename,
                        "x": full_x,
                        "y": full_y,
                        "level": level,
                        "tissue_percentage": tissue_percentage,
                        "patch_index": count,
                    }
                    metadata["patches"].append(patch_info)

                count += 1

            # Draw debug visualization
            if create_debug_images:
                rect = [
                    thumb_x,
                    thumb_y,
                    thumb_x + thumb_patch_size,
                    thumb_y + thumb_patch_size,
                ]
                draw.rectangle(
                    rect, outline="green" if should_infer else "red", width=1
                )

        print(f"Extracted {count} patches from {wsi_path} after {attempts} attempts")

    else:  # contiguous mode
        print(f"Extracting contiguous patches from tissue regions")

        # Add at beginning of the contiguous mode section
        max_patches_to_extract = num_patches

        # Calculate step size (with overlap)
        step_size = int(patch_size * (1 - overlap))

        # Calculate number of patches in each dimension
        num_patches_x = (width - patch_size) // step_size + 1
        num_patches_y = (height - patch_size) // step_size + 1

        print(
            f"Creating {num_patches_x}x{num_patches_y} grid of patches with step size {step_size}"
        )

        # Add this in the contiguous mode section, right after calculating num_patches_x and num_patches_y
        if create_debug_images:
            print(
                f"Full grid would be {num_patches_x}x{num_patches_y} = {num_patches_x*num_patches_y} patches"
            )

        # Create a dilated tissue mask to capture regions near tissue
        dilated_tissue_mask = tissue_mask.copy()
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_tissue_mask = cv2.dilate(
            dilated_tissue_mask.astype(np.uint8), dilation_kernel, iterations=1
        )

        # Create a grid of ROIs using the tissue mask as a filter
        roi_positions = []
        for y_idx in range(num_patches_y):
            for x_idx in range(num_patches_x):
                # Calculate thumbnail coordinates
                thumb_x = int(x_idx * step_size * scale_factor)
                thumb_y = int(y_idx * step_size * scale_factor)

                # Skip if outside thumbnail bounds
                if (
                    thumb_x + thumb_patch_size >= thumbnail_np.shape[1]
                    or thumb_y + thumb_patch_size >= thumbnail_np.shape[0]
                ):
                    continue

                # Check if this patch overlaps with the dilated tissue mask
                patch_mask = dilated_tissue_mask[
                    thumb_y : thumb_y + thumb_patch_size,
                    thumb_x : thumb_x + thumb_patch_size,
                ]
                if np.sum(patch_mask) > 0:  # If any tissue pixel exists in this patch
                    roi_positions.append((x_idx, y_idx, thumb_x, thumb_y))

        print(f"Found {len(roi_positions)} potential ROIs after tissue mask filtering")

        # Now iterate only through the filtered ROI positions
        for x_idx, y_idx, thumb_x, thumb_y in roi_positions:
            # Calculate full resolution coordinates
            full_x = x_idx * step_size
            full_y = y_idx * step_size

            # Check tissue percentage in thumbnail
            region = tissue_mask[
                thumb_y : thumb_y + thumb_patch_size,
                thumb_x : thumb_x + thumb_patch_size,
            ]
            tissue_percentage = np.sum(region) / region.size

            if tissue_percentage <= tissue_threshold:
                if create_debug_images:
                    rect = [
                        thumb_x,
                        thumb_y,
                        thumb_x + thumb_patch_size,
                        thumb_y + thumb_patch_size,
                    ]
                    draw.rectangle(rect, outline="yellow", width=1)
                continue

            # Check exclusion conditions on base-level coordinates
            should_exclude = False
            satisfied_conditions = []

            for condition in exclusion_conditions:
                coord, operator, value = condition

                # Get the appropriate coordinate value
                if coord.lower() == "x":
                    coord_value = full_x
                elif coord.lower() == "y":
                    coord_value = full_y
                else:
                    continue

                # Apply the operator comparison
                condition_met = False
                if operator == "<" and coord_value < value:
                    condition_met = True
                elif operator == ">" and coord_value > value:
                    condition_met = True
                elif operator == "<=" and coord_value <= value:
                    condition_met = True
                elif operator == ">=" and coord_value >= value:
                    condition_met = True
                elif operator == "==" and coord_value == value:
                    condition_met = True

                if condition_met:
                    satisfied_conditions.append(True)
                else:
                    satisfied_conditions.append(False)

            # Determine exclusion based on mode
            if exclusion_mode == "any" and any(satisfied_conditions):
                should_exclude = True
            elif (
                exclusion_mode == "all"
                and all(satisfied_conditions)
                and satisfied_conditions
            ):
                should_exclude = True

            if should_exclude:
                if create_debug_images and len(satisfied_conditions) > 0:
                    # Draw excluded regions in blue
                    rect = [
                        thumb_x,
                        thumb_y,
                        thumb_x + thumb_patch_size,
                        thumb_y + thumb_patch_size,
                    ]
                    draw.rectangle(rect, outline="blue", width=1)
                continue

            # Extract full resolution patch
            patch_pil = slide.read_region(
                (full_x, full_y), level, (patch_size, patch_size)
            ).convert("RGB")
            patch_np = np.array(patch_pil)

            # Final verification on the full resolution patch
            should_infer = is_tissue_patch(patch_np, tissue_threshold)

            if should_infer:
                # Store patch with coordinates
                patches.append((patch_np, full_x, full_y))
                if (
                    count % 100 == 0
                ):  # Print every 100 patches to avoid flooding console
                    print(
                        f"Patch {count} at position x={full_x}, y={full_y} is a tissue patch"
                    )

                # Save patch if requested
                if save_patches:
                    patch_filename = f"{slide_name}_x{full_x}_y{full_y}_l{level}.png"
                    patch_path = os.path.join(slide_output_dir, patch_filename)
                    patch_pil.save(patch_path)

                    # Store metadata
                    patch_info = {
                        "filename": patch_filename,
                        "x": full_x,
                        "y": full_y,
                        "level": level,
                        "tissue_percentage": tissue_percentage,
                        "patch_index": count,
                    }
                    metadata["patches"].append(patch_info)

                count += 1

            # Draw debug visualization
            if create_debug_images:
                rect = [
                    thumb_x,
                    thumb_y,
                    thumb_x + thumb_patch_size,
                    thumb_y + thumb_patch_size,
                ]
                draw.rectangle(
                    rect, outline="green" if should_infer else "red", width=1
                )

            # Then add this check inside the loop
            if count >= max_patches_to_extract:
                break

    # Save debug overlay
    if create_debug_images:
        debug_path = os.path.join(debug_output_dir, "patch_debug_overlay.png")
        downsampled.save(debug_path)

    print(f"Extracted {count} patches from {wsi_path}")
    slide.close()

    # Save metadata if patches were saved
    if save_patches:
        metadata_path = os.path.join(slide_output_dir, f"{slide_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved {count} patches and metadata to {slide_output_dir}")

        return patches, metadata

    return patches


def parse_exclusions(exclusion_str):
    """Parse exclusion conditions string into a list of tuples."""
    if not exclusion_str:
        return []
    
    conditions = []
    for condition in exclusion_str.split(','):
        parts = condition.strip().split(':')
        if len(parts) != 3:
            print(f"Warning: Invalid exclusion condition '{condition}', skipping")
            continue
            
        coord, operator, value = parts
        if operator not in ['<', '>', '<=', '>=', '==']:
            print(f"Warning: Invalid operator '{operator}' in '{condition}', skipping")
            continue
            
        try:
            value = int(value)
        except ValueError:
            print(f"Warning: Invalid value '{value}' in '{condition}', skipping")
            continue
            
        conditions.append((coord, operator, value))
        
    return conditions


def main():
    """Parse command line arguments and run the patch extractor."""
    parser = argparse.ArgumentParser(description="Extract patches from a whole slide image.")
    
    # Required arguments
    parser.add_argument("--input", "-i", required=True, help="Path to the input SVS file")
    
    # Extraction parameters
    parser.add_argument("--patch-size", "-p", type=int, default=256, 
                      help="Size of patches to extract (default: 256)")
    parser.add_argument("--overlap", "-o", type=float, default=0.25, 
                      help="Overlap between patches, 0-1 (default: 0.25)")
    parser.add_argument("--level", "-l", type=int, default=0, 
                      help="WSI pyramid level to extract from (default: 0)")
    parser.add_argument("--tissue-threshold", "-t", type=float, default=0.05, 
                      help="Minimum tissue percentage threshold (default: 0.05)")
    parser.add_argument("--num-patches", "-n", type=int, default=1000, 
                      help="Maximum number of patches to extract (default: 1000)")
    
    # Debug options
    parser.add_argument("--debug", "-d", action="store_true", 
                      help="Save debug outputs")
    parser.add_argument("--debug-output-dir", "-do", default="debug_output", 
                      help="Directory to save debug outputs (default: 'debug_output')")
    
    # Exclusion conditions
    parser.add_argument("--exclusions", "-e", type=str, 
                      help="Comma-separated exclusion conditions in format 'coord:operator:value'. "
                           "Example: 'x:>:1000,y:<:500' excludes x>1000 and y<500")
    parser.add_argument("--exclusion-mode", "-em", choices=["any", "all"], default="any", 
                      help="'any' to exclude if any condition is met, 'all' for all conditions (default: 'any')")
    
    # Extraction mode
    parser.add_argument("--mode", "-m", choices=["random", "contiguous"], default="random", 
                      help="Extraction mode: 'random' or 'contiguous' (default: 'random')")
    
    # Output options
    parser.add_argument("--save-patches", "-s", action="store_true", 
                      help="Save extracted patches to disk")
    parser.add_argument("--output-dir", "-od", default="extracted_patches", 
                      help="Directory to save extracted patches (default: 'extracted_patches')")
    parser.add_argument("--label", help="Optional label/class for organizing patches")
    
    args = parser.parse_args()
    
    # Parse exclusion conditions
    exclusion_conditions = parse_exclusions(args.exclusions)
    
    # Extract patches
    result = extract_patches_from_wsi(
        wsi_path=args.input,
        patch_size=args.patch_size,
        overlap=args.overlap,
        level=args.level,
        tissue_threshold=args.tissue_threshold,
        create_debug_images=args.debug,
        debug_output_dir=args.debug_output_dir,
        num_patches=args.num_patches,
        exclusion_conditions=exclusion_conditions,
        exclusion_mode=args.exclusion_mode,
        extraction_mode=args.mode,
        save_patches=args.save_patches,
        output_dir=args.output_dir,
        label=args.label,
    )
    
    # Handle result based on whether patches were saved
    if isinstance(result, tuple):
        patches, metadata = result
        print(f"Extracted and saved {len(patches)} patches with metadata")
    else:
        patches = result
        print(f"Extracted {len(patches)} patches (not saved to disk)")
        
        # Save patches to output_dir if not already saved but patches were extracted
        if patches and not args.save_patches and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            slide_name = os.path.splitext(os.path.basename(args.input))[0]
            
            print(f"Saving {len(patches)} patches to {args.output_dir}")
            for i, (patch_np, x, y) in enumerate(patches):
                patch_filename = f"{slide_name}_x{x}_y{y}_l{args.level}.png"
                patch_path = os.path.join(args.output_dir, patch_filename)
                patch_pil = Image.fromarray(patch_np)
                patch_pil.save(patch_path)
                
                if i % 100 == 0:
                    print(f"Saved {i}/{len(patches)} patches")
                    
            print(f"Successfully saved {len(patches)} patches to {args.output_dir}")


if __name__ == "__main__":
    main()