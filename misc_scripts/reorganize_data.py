import os
import shutil
from pathlib import Path

def create_directory_structure(base_dir):
    """Create the new directory structure"""
    for split in ['train', 'val', 'test']:
        for stain in ['ihc', 'pas']:
            os.makedirs(os.path.join(base_dir, split, stain), exist_ok=True)

def copy_data(source_dir, target_dir):
    """Copy all files from source to target directory"""
    for file in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, file)
        if os.path.isfile(source_path):
            shutil.copy2(source_path, target_path)

def main():
    # Base directories
    base_source_dir = "/data2/monkey-challenge/data"
    base_target_dir = "data"  # New structure will be created here
    
    # Create new directory structure
    create_directory_structure(base_target_dir)
    
    # Copy training data
    print("Copying training data...")
    train_pas_source = os.path.join(base_source_dir, "train_cpg_patches_128_overlap_5um_label", "pas_patches")
    train_ihc_source = os.path.join(base_source_dir, "train_cpg_patches_128_overlap_5um_label", "ihc_patches")
    
    copy_data(train_pas_source, os.path.join(base_target_dir, "train", "pas"))
    copy_data(train_ihc_source, os.path.join(base_target_dir, "train", "ihc"))
    
    # Copy test data to both val and test directories
    print("Copying test data to val and test directories...")
    test_pas_source = os.path.join(base_source_dir, "test_cpg_patches_128_overlap_5um_label", "pas_patches")
    test_ihc_source = os.path.join(base_source_dir, "test_cpg_patches_128_overlap_5um_label", "ihc_patches")
    
    # Copy to val directory
    copy_data(test_pas_source, os.path.join(base_target_dir, "val", "pas"))
    copy_data(test_ihc_source, os.path.join(base_target_dir, "val", "ihc"))
    
    # Copy to test directory
    copy_data(test_pas_source, os.path.join(base_target_dir, "test", "pas"))
    copy_data(test_ihc_source, os.path.join(base_target_dir, "test", "ihc"))
    
    print("Data reorganization complete!")

if __name__ == "__main__":
    main() 