"""
Quantitative analysis of 3D tomography images of porous materials.

This module provides functions to analyze 3D CT images of porous materials,
calculating porosity, autocorrelations, and cross-correlations.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict


def calculate_porosity(image_3d: np.ndarray, pore_value: int = 1) -> float:
    """
    Calculate porosity of a 3D image.
    
    Porosity is defined as the ratio of pore voxels to total voxels.
    
    Parameters:
    -----------
    image_3d : np.ndarray
        3D binary image where pore_value indicates pore phase
    pore_value : int
        Value representing pore phase (default: 1)
    
    Returns:
    --------
    float
        Porosity value between 0 and 1
    """
    total_voxels = image_3d.size
    pore_voxels = np.sum(image_3d == pore_value)
    porosity = pore_voxels / total_voxels
    return porosity


def calculate_autocorrelation(image_3d: np.ndarray, phase_value: int = 1, 
                              normalize: bool = True) -> np.ndarray:
    """
    Calculate 3D autocorrelation of a specific phase.
    
    The autocorrelation function describes the spatial correlation of the phase
    with itself at different distances.
    
    Parameters:
    -----------
    image_3d : np.ndarray
        3D binary image
    phase_value : int
        Value representing the phase of interest (default: 1)
    normalize : bool
        Whether to normalize the autocorrelation (default: True)
    
    Returns:
    --------
    np.ndarray
        3D autocorrelation function
    """
    # Create binary mask for the specific phase
    phase_mask = (image_3d == phase_value).astype(float)
    
    # Calculate autocorrelation using FFT-based convolution
    autocorr = signal.fftconvolve(phase_mask, phase_mask[::-1, ::-1, ::-1], 
                                  mode='same')
    
    if normalize:
        # Normalize by the variance at zero lag
        center = tuple(s // 2 for s in autocorr.shape)
        autocorr = autocorr / autocorr[center]
    
    return autocorr


def calculate_crosscorrelation(image_3d: np.ndarray, phase1_value: int = 1,
                               phase2_value: int = 0, normalize: bool = True) -> np.ndarray:
    """
    Calculate 3D cross-correlation between two phases.
    
    The cross-correlation describes the spatial relationship between two phases.
    
    Parameters:
    -----------
    image_3d : np.ndarray
        3D binary image
    phase1_value : int
        Value representing the first phase (default: 1, pore)
    phase2_value : int
        Value representing the second phase (default: 0, solid)
    normalize : bool
        Whether to normalize the cross-correlation (default: True)
    
    Returns:
    --------
    np.ndarray
        3D cross-correlation function
    """
    # Create binary masks for both phases
    phase1_mask = (image_3d == phase1_value).astype(float)
    phase2_mask = (image_3d == phase2_value).astype(float)
    
    # Calculate cross-correlation using FFT-based convolution
    crosscorr = signal.fftconvolve(phase1_mask, phase2_mask[::-1, ::-1, ::-1],
                                   mode='same')
    
    if normalize:
        # Normalize by geometric mean of variances
        auto1 = signal.fftconvolve(phase1_mask, phase1_mask[::-1, ::-1, ::-1],
                                   mode='same')
        auto2 = signal.fftconvolve(phase2_mask, phase2_mask[::-1, ::-1, ::-1],
                                   mode='same')
        center = tuple(s // 2 for s in crosscorr.shape)
        norm_factor = np.sqrt(auto1[center] * auto2[center])
        if norm_factor > 0:
            crosscorr = crosscorr / norm_factor
    
    return crosscorr


def extract_radial_profile(correlation_3d: np.ndarray, max_distance: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 1D radial profile from 3D correlation function.
    
    This averages the correlation values at each radial distance from the center.
    
    Parameters:
    -----------
    correlation_3d : np.ndarray
        3D correlation function
    max_distance : int
        Maximum distance to extract (default: min dimension / 2)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        distances, radial_profile
    """
    shape = correlation_3d.shape
    center = np.array([s // 2 for s in shape])
    
    if max_distance is None:
        max_distance = min(shape) // 2
    
    # Create distance map from center
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance_map = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    
    # Bin by distance
    distances = np.arange(0, max_distance)
    radial_profile = np.zeros(max_distance)
    
    for i, dist in enumerate(distances):
        mask = (distance_map >= dist) & (distance_map < dist + 1)
        if np.any(mask):
            radial_profile[i] = np.mean(correlation_3d[mask])
    
    return distances, radial_profile


def analyze_sample(image_3d: np.ndarray, pore_value: int = 1, solid_value: int = 0,
                  max_corr_distance: int = 50) -> Dict[str, np.ndarray]:
    """
    Perform complete analysis on a single 3D sample.
    
    Parameters:
    -----------
    image_3d : np.ndarray
        3D binary image (200x200x200)
    pore_value : int
        Value representing pore phase
    solid_value : int
        Value representing solid phase
    max_corr_distance : int
        Maximum distance for correlation profiles
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing porosity and correlation profiles
    """
    # Calculate porosity
    porosity = calculate_porosity(image_3d, pore_value)
    
    # Calculate autocorrelations
    pore_autocorr = calculate_autocorrelation(image_3d, pore_value)
    solid_autocorr = calculate_autocorrelation(image_3d, solid_value)
    
    # Calculate cross-correlation
    crosscorr = calculate_crosscorrelation(image_3d, pore_value, solid_value)
    
    # Extract radial profiles
    _, pore_profile = extract_radial_profile(pore_autocorr, max_corr_distance)
    _, solid_profile = extract_radial_profile(solid_autocorr, max_corr_distance)
    _, cross_profile = extract_radial_profile(crosscorr, max_corr_distance)
    
    return {
        'porosity': porosity,
        'pore_autocorrelation': pore_profile,
        'solid_autocorrelation': solid_profile,
        'crosscorrelation': cross_profile
    }
