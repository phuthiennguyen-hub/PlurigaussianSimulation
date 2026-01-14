"""
Main script to process 197 3D CT images and generate CSV output with analysis results.

This script processes all sample images and outputs a CSV file containing:
- Porosity
- Autocorrelation of pore phase
- Autocorrelation of solid phase
- Cross-correlation of two phases

Usage:
    python process_samples.py --input_dir <path_to_samples> --output_csv <output_file.csv>
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import tifffile
from tomography_analysis import analyze_sample


def load_sample(file_path: str) -> np.ndarray:
    """
    Load a 3D CT image from file.
    
    Supports various formats including TIFF stacks, NumPy arrays, etc.
    
    Parameters:
    -----------
    file_path : str
        Path to the image file
    
    Returns:
    --------
    np.ndarray
        3D image array
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in ['.tif', '.tiff']:
        # Load TIFF stack
        image = tifffile.imread(file_path)
    elif file_ext in ['.npy']:
        # Load NumPy array
        image = np.load(file_path)
    elif file_ext in ['.npz']:
        # Load compressed NumPy array
        data = np.load(file_path)
        # Assume the array is stored with key 'image' or take the first array
        if 'image' in data:
            image = data['image']
        else:
            image = data[list(data.keys())[0]]
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return image


def process_all_samples(input_dir: str, max_corr_distance: int = 50,
                       file_pattern: str = "*.tif") -> pd.DataFrame:
    """
    Process all samples in the input directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing sample files
    max_corr_distance : int
        Maximum distance for correlation profiles
    file_pattern : str
        Pattern to match sample files (default: "*.tif")
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all analysis results
    """
    input_path = Path(input_dir)
    
    # Find all matching files
    sample_files = sorted(input_path.glob(file_pattern))
    
    if not sample_files:
        # Try other common formats
        for pattern in ["*.npy", "*.npz", "*.tiff"]:
            sample_files = sorted(input_path.glob(pattern))
            if sample_files:
                break
    
    if not sample_files:
        raise FileNotFoundError(f"No sample files found in {input_dir}")
    
    print(f"Found {len(sample_files)} samples to process")
    
    # Initialize results storage
    results = []
    
    for i, sample_file in enumerate(sample_files):
        print(f"Processing sample {i+1}/{len(sample_files)}: {sample_file.name}")
        
        try:
            # Load the sample
            image_3d = load_sample(str(sample_file))
            
            # Verify dimensions
            if image_3d.shape != (200, 200, 200):
                print(f"  Warning: Sample {sample_file.name} has shape {image_3d.shape}, expected (200, 200, 200)")
            
            # Analyze the sample
            analysis = analyze_sample(image_3d, max_corr_distance=max_corr_distance)
            
            # Prepare result row
            result_row = {
                'sample_id': i + 1,
                'sample_name': sample_file.stem,
                'porosity': analysis['porosity']
            }
            
            # Add correlation values at different distances
            # Store correlation profiles as separate columns
            for dist in range(max_corr_distance):
                result_row[f'pore_autocorr_dist_{dist}'] = analysis['pore_autocorrelation'][dist]
                result_row[f'solid_autocorr_dist_{dist}'] = analysis['solid_autocorrelation'][dist]
                result_row[f'crosscorr_dist_{dist}'] = analysis['crosscorrelation'][dist]
            
            results.append(result_row)
            
        except Exception as e:
            print(f"  Error processing {sample_file.name}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Process 3D CT images and calculate porosity and correlations'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing the 3D CT image samples'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='analysis_results.csv',
        help='Output CSV file path (default: analysis_results.csv)'
    )
    parser.add_argument(
        '--max_corr_distance',
        type=int,
        default=50,
        help='Maximum distance for correlation analysis (default: 50)'
    )
    parser.add_argument(
        '--file_pattern',
        type=str,
        default='*.tif',
        help='File pattern to match samples (default: *.tif)'
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    print("=" * 60)
    print("3D Tomography Image Analysis")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Max correlation distance: {args.max_corr_distance}")
    print("=" * 60)
    
    # Process all samples
    results_df = process_all_samples(
        args.input_dir,
        max_corr_distance=args.max_corr_distance,
        file_pattern=args.file_pattern
    )
    
    # Save to CSV
    results_df.to_csv(args.output_csv, index=False)
    
    print("=" * 60)
    print(f"Analysis complete! Results saved to {args.output_csv}")
    print(f"Processed {len(results_df)} samples")
    
    if len(results_df) > 0:
        print(f"Average porosity: {results_df['porosity'].mean():.4f}")
    else:
        print("Warning: No samples were successfully processed")
    
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
