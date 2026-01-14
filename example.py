"""
Example script demonstrating usage of the tomography analysis tools.

This script creates synthetic 3D porous material samples and processes them
to demonstrate the analysis workflow.
"""

import numpy as np
import os
from pathlib import Path
from tomography_analysis import analyze_sample
import pandas as pd


def generate_synthetic_sample(size=(200, 200, 200), porosity=0.3, seed=None):
    """
    Generate a synthetic 3D porous material sample.
    
    Parameters:
    -----------
    size : tuple
        Dimensions of the 3D image
    porosity : float
        Target porosity (approximate)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    np.ndarray
        3D binary image (1=pore, 0=solid)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random field
    random_field = np.random.randn(*size)
    
    # Apply Gaussian smoothing to create correlated structure
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(random_field, sigma=5.0)
    
    # Threshold to achieve target porosity
    threshold = np.percentile(smoothed, (1 - porosity) * 100)
    binary_image = (smoothed > threshold).astype(np.uint8)
    
    return binary_image


def main():
    """Main example demonstration."""
    print("=" * 70)
    print("3D Tomography Analysis - Example Demonstration")
    print("=" * 70)
    
    # Create output directory for examples
    output_dir = Path("example_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate a few synthetic samples
    num_samples = 5
    print(f"\nGenerating {num_samples} synthetic samples...")
    
    samples_dir = output_dir / "synthetic_samples"
    samples_dir.mkdir(exist_ok=True)
    
    for i in range(num_samples):
        # Generate sample with varying porosity
        target_porosity = 0.2 + i * 0.1  # 0.2, 0.3, 0.4, 0.5, 0.6
        sample = generate_synthetic_sample(porosity=target_porosity, seed=i)
        
        # Save as numpy array
        sample_path = samples_dir / f"sample_{i+1:03d}.npy"
        np.save(sample_path, sample)
        print(f"  Created {sample_path.name} (target porosity: {target_porosity:.2f})")
    
    # Analyze the samples
    print("\n" + "=" * 70)
    print("Analyzing samples...")
    print("=" * 70)
    
    results = []
    for i in range(num_samples):
        sample_path = samples_dir / f"sample_{i+1:03d}.npy"
        print(f"\nProcessing {sample_path.name}...")
        
        # Load sample
        image_3d = np.load(sample_path)
        
        # Analyze
        analysis = analyze_sample(image_3d, max_corr_distance=50)
        
        # Print results
        print(f"  Porosity: {analysis['porosity']:.4f}")
        print(f"  Pore autocorr at distance 0: {analysis['pore_autocorrelation'][0]:.4f}")
        print(f"  Pore autocorr at distance 10: {analysis['pore_autocorrelation'][10]:.4f}")
        print(f"  Solid autocorr at distance 0: {analysis['solid_autocorrelation'][0]:.4f}")
        print(f"  Cross-correlation at distance 0: {analysis['crosscorrelation'][0]:.4f}")
        
        # Store results
        result_row = {
            'sample_id': i + 1,
            'sample_name': sample_path.stem,
            'porosity': analysis['porosity']
        }
        
        # Add first 20 correlation values as example
        for dist in range(20):
            result_row[f'pore_autocorr_dist_{dist}'] = analysis['pore_autocorrelation'][dist]
            result_row[f'solid_autocorr_dist_{dist}'] = analysis['solid_autocorrelation'][dist]
            result_row[f'crosscorr_dist_{dist}'] = analysis['crosscorrelation'][dist]
        
        results.append(result_row)
    
    # Save results to CSV
    print("\n" + "=" * 70)
    print("Saving results to CSV...")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    output_csv = output_dir / "example_analysis_results.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"\nResults saved to: {output_csv}")
    print(f"Number of samples processed: {len(df)}")
    print(f"\nSummary statistics:")
    print(f"  Mean porosity: {df['porosity'].mean():.4f}")
    print(f"  Std porosity: {df['porosity'].std():.4f}")
    print(f"  Min porosity: {df['porosity'].min():.4f}")
    print(f"  Max porosity: {df['porosity'].max():.4f}")
    
    print("\n" + "=" * 70)
    print("Example demonstration complete!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - Synthetic samples: {samples_dir}")
    print(f"  - Analysis results: {output_csv}")
    print("\nYou can now use process_samples.py with real CT image data:")
    print(f"  python process_samples.py --input_dir <your_data_dir> --output_csv results.csv")
    print("=" * 70)


if __name__ == '__main__':
    main()
