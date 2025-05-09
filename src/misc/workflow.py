#!/usr/bin/env python3
import os
import re
import subprocess
import shutil
import glob
from pathlib import Path

def parse_line(line):
    # Skip empty lines or batch headers
    if not line.strip() or line.startswith('batch_'):
        return None
        
    # Extract filename
    filename_match = re.search(r'(anon_[a-f0-9-]+\.svs)', line)
    if not filename_match:
        return None
        
    filename = filename_match.group(1)
    
    # Extract stain type and split
    stain_match = re.search(r'\((IHC|EVG|PAS|HnE|Silver|ihc|hne|pas|silver|evg)(?:\?)?\s*-\s*(training|validation|test|train|val|testing)', line, re.IGNORECASE)
    if not stain_match:
        return None
        
    stain = stain_match.group(1).lower()
    split = stain_match.group(2).lower()
    
    if split == 'training':
        split = 'train'
    elif split == 'validation':
        split = 'val'
    elif split == 'testing':
        split = 'test'
        
    # Skip if stain has question mark
    if '?' in line:
        return None
        
    # Skip if not train/val
    if split not in ['train', 'val']:
        return None
        
    # Extract number of patches
    num_patches = 1000  # default
    k_match = re.search(r'(\d+)k', line)
    if k_match:
        num_patches = int(k_match.group(1)) * 1000
    
    # Extract exclusion coordinates - handle both patterns
    # Try both patterns: with commas and without commas
    coords_match = re.search(r'\((\d+)\s*(?:,\s*)?(\d+)\s*(?:,\s*)?(\d+)\s*(?:,\s*)?(\d+)\)', line)
    
    if coords_match:
        x_gt, y_lt, x_lt, y_gt = map(int, coords_match.groups())
        exclusions = f"x:>:{x_gt},y:<:{y_lt},x:<:{x_lt},y:>:{y_gt}"
    else:
        # No valid exclusion coordinates found
        print(f"Warning: No valid exclusion coordinates found for {filename}")
        return None
    
    return {
        'filename': filename,
        'stain': stain,
        'split': split,
        'num_patches': num_patches,
        'exclusions': exclusions
    }

def find_svs_file(filename):
    """Find the SVS file in the base directory"""
    # Start a subprocess to find the file
    cmd = ["find", "/vol/biomedic3/histopatho/win_share", "-name", filename]
    print(f"Searching for file with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        paths = result.stdout.strip().split('\n')
        
        if paths and paths[0]:
            print(f"Found SVS file at: {paths[0]}")
            return paths[0]
        
        print(f"SVS file not found: {filename}")
        return None
    except subprocess.TimeoutExpired:
        print(f"Search timed out for {filename}")
        return None

def process_file(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        result = parse_line(line)
        if not result:
            continue
            
        print(f"\nProcessing line {line_num}: {result['filename']}")
        print(f"Stain: {result['stain']}")
        print(f"Split: {result['split']}")
        print(f"Number of patches: {result['num_patches']}")
        print(f"Exclusions: {result['exclusions']}")
        
        # Find the SVS file
        svs_path = find_svs_file(result['filename'])
        if not svs_path:
            print(f"WARNING: Could not find SVS file {result['filename']}, skipping")
            continue
        
        # Create tmp directory
        tmp_dir = './data/tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Create debug output directory
        debug_dir = './debug_output'
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create command (removed --save-patches and --label flags as these are handled after extraction)
        cmd = [
            'python', 'src/misc/patch_extractor.py',
            '--input', svs_path,
            '--patch-size', '256',
            '--overlap', '0.25',
            '--level', '0',
            '--tissue-threshold', '0.20',
            '--debug',
            '--debug-output-dir', debug_dir,
            '--exclusions', result['exclusions'],
            '--exclusion-mode', 'any',
            '--mode', 'random',
            '--num-patches', str(result['num_patches']),
            '--output-dir', tmp_dir
        ]
        
        # Print the command
        print("\nRunning command:")
        print(' '.join(cmd))
        
        # Run patch extractor
        process = subprocess.Popen(cmd)
        
        # Wait for user input
        input("\nReview debug output and press Enter to continue when extraction completes...")
        
        # Check if process is still running
        if process.poll() is None:
            print("Process still running, waiting for it to complete...")
            process.wait()
        
        # Wait for user input before moving files
        input("\nPress Enter to move files to final destination...")
        
        # Create destination directory
        dest_dir = f"data/inhouse/{result['split']}/{result['stain']}"
        os.makedirs(dest_dir, exist_ok=True)
        
        # Move files from tmp to final destination
        # With the removal of --save-patches and --label, patches are saved directly in tmp_dir
        # instead of in label-specific subdirectories
        moved = 0
        slide_name = os.path.splitext(os.path.basename(result['filename']))[0]
        
        # Move all files related to this slide to destination
        for item in glob.glob(f"{tmp_dir}/*{slide_name}*", recursive=True):
            if os.path.isfile(item):
                filename = os.path.basename(item)
                dst = os.path.join(dest_dir, filename)
                shutil.move(item, dst)
                moved += 1
                
        print(f"Moved {moved} files to {dest_dir}")
        
        # Clean tmp directory
        for item in glob.glob(f"{tmp_dir}/*"):
            if os.path.isdir(item):
                shutil.rmtree(item)
        
        print(f"Cleaned temporary directory")

if __name__ == "__main__":
    process_file('data/inhouse/new_cases.txt')