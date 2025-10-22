#!/usr/bin/env python3
"""
Batch process all subfolders and record grasp videos
"""
import os
import subprocess
import sys
from pathlib import Path

def batch_record(base_dir, max_record=3):
    """
    Process all subfolders in base_dir and record grasp videos
    
    Args:
        base_dir: Directory containing run folders
        max_record: Number of grasps to record per folder
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: {base_dir} does not exist")
        return
    
    # Create output directory inside base_dir
    output_base = base_path / "1_recordings"
    output_base.mkdir(exist_ok=True)
    
    # Get all subdirectories
    folders = [f for f in base_path.iterdir() if f.is_dir() and f.name != "1_recordings"]
    
    print(f"Found {len(folders)} folders to process")
    print(f"Recording {max_record} grasps per folder")
    print(f"Output will be saved to: {output_base}/")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    for i, folder in enumerate(folders, 1):
        folder_name = folder.name
        print(f"\n[{i}/{len(folders)}] Processing {folder_name}...")
        
        # Create output directory for this folder
        output_dir = output_base / folder_name
        print(f"Output directory: {output_dir}")
        # Build command
        cmd = [
            "mjpython", "-m", "visualization.replay",
            "-r", str(folder),
            "-fs", "-rb", "-s",
            "--record",
            "--max-record", str(max_record),
            "--output-dir", str(output_dir)
        ]
        
        try:
            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✓ Completed {folder_name}")
                success_count += 1
            else:
                print(f"✗ Failed {folder_name}")
                print(f"Error: {result.stderr[:200]}")
                fail_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout {folder_name}")
            fail_count += 1
        except Exception as e:
            print(f"✗ Error {folder_name}: {e}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"Success: {success_count}/{len(folders)}")
    print(f"Failed: {fail_count}/{len(folders)}")
    print(f"Videos saved in: {output_base}/")

if __name__ == "__main__":
    # Default settings
    base_dir = "ycb_chips_can_2025-10-13_07-50"
    max_record = 3
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    if len(sys.argv) > 2:
        max_record = int(sys.argv[2])
    
    batch_record(base_dir, max_record)