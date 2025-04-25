#!/bin/bash

# Script to find .svs files in /vol/biomedic3/histopatho/win_share based on IDs in file_ids.txt

# Output file to store results
output_file="found_svs_files.txt"
> "$output_file"  # Clear output file if it exists

echo "Searching for SVS files in /vol/biomedic3/histopatho/win_share..."
echo "Results will be saved in $output_file"

# Read each line from file_ids.txt
while IFS= read -r file_id || [[ -n "$file_id" ]]; do
  # Skip empty lines
  if [[ -z "$file_id" ]]; then
    continue
  fi
  
  echo "Searching for $file_id.svs"
  
  # Use find command to locate the file
  result=$(find /vol/biomedic3/histopatho/win_share -type f -name "$file_id.svs" 2>/dev/null)
  
  if [[ -n "$result" ]]; then
    echo "Found: $file_id → $result"
    echo "$result" >> "$output_file"
  else
    echo "Not found: $file_id"
    echo "$file_id NOT FOUND" >> "$output_file"
  fi
done < file_ids.txt

echo "Search completed. Results saved in $output_file" 