from ast import If
import numpy as np
import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage import filters, morphology
import matplotlib.pyplot as plt
import os
import datetime


###############################################################
#                                                             # 
#  DEGRID 1.0 - Detection and Elimination of GRID artifacts   #
#                                                             #
###############################################################
"""
Grid artifact removal in single-grid dark-field X-ray imaging.

This system implements automatic detection and removal of grid 
artifacts in single-grid dark-field X-ray imaging using Fourier 
space analysis and L-BFGS-B optimization.

Date: 18.08.2025
Author: Werneri A. Lindberg
Acknowledgment: Some parts of the script were written with assistance from GitHub Copilot.

Please check the requirements.txt file for the complete list of dependencies and their versions.
Python           3.12.0
opencv           4.11.0.86
matplotlib       3.10.3
numpy            2.3.1
pillow           11.3.0
scipy            1.16.0
scikit-image     0.25.2

"""

# ==================================================================================
# DESCRIPTION:
# > Select via file dialog: the dark-field image with grid artifact, 
#   the reference image with only the grid artifact,
#   and the clean flat field image without grid.
# > Results are saved in the same folder as the input images.
# > Log file is saved with all parameters.
# ==================================================================================



# ============= PARAMETERS ====================================
class ProcessingParams:

    def __init__(self):
        # ===============================================================================
        # OPTIMIZATION CONTROL
        # ===============================================================================
        
        self.optimization_method = 'L-BFGS-B'
        self.max_iterations = 100              # Maximum iterations

        self.ftol = 1e-6                       # Function tolerance for convergence. 
                                               # If quality score is tiny, we assume we've found the best answer.

        self.gtol = 1e-6                       # Gradient tolerance for convergence.
                                               # If the "slope" is almost flat, we assume we've found the best answer.
        
        self.tolerance = 1e-6                  # General convergence tolerance.
                                               # This is a catch-all rule for stopping the optimization process
                                               # when all changes (in parameters or score) are very small.
        
        # ===============================================================================


        # ===============================================================================
        # GRID REMOVAL PARAMETERS
        # ===============================================================================
        #
        # Initial guesses for grid parameters
        # ------------------------------
        self.base_filter_radius = 1         # Starting filter radius for notch filters in frequency domain.
                                            # Used as the lower bound in adaptive filter size selection.
        
        self.max_filter_radius = 20         # Maximum filter radius to test

        self.default_filter_radius = 5      # Default filter radius for grid removal if adaptive selection is not used.
                                            # Acts as a safe fallback value.

        self.quality_threshold = 0.99       # Quality threshold for early stopping during adaptive filtering.
                                            # If the grid removal quality (measured by MSE) exceeds this value,
                                            # the algorithm will stop searching for better filter sizes.
        #
        # NOTE: These parameters are used as initial values and bounds for the adaptive, automated grid removal.
        # The algorithm analyzes the image data and automatically selects the optimal filter radius.
        # In other words, these are not arbitrary constants and are used for the data-driven optimization.

        # Quality evaluation (MSE)
        # ------------------------
        self.confidence_peak_threshold = 3      # Minimum number of valid grid peaks required for full confidence in the quality metric.
                                                # This value is set low to ensure robust operation even in images with few grid peaks,
                                                # and is based on observations with typical X-ray grid/mesh reference images.
        

        # Grid Pattern Characteristics
        # ----------------------------
        self.expected_grid_frequency_range = (0.01, 0.3)  # Frequency range for typical X-ray grids [fraction of the max frequency]
      

        self.exclude_dc_radius = 20                       # Radius (in pixels) around the DC component (zero-frequency) to exclude in the FFT,
                                                          # which is not considered part of the grid pattern. This is a setup-specific 
                                                          # calibration parameter: For a given imaging system (fixed source, detector, and grid/mesh), 
                                                          # the central low-frequency region in the FFT contains dominant non-grid content.
                                                          # The exclusion radius should be determined once during system calibration 
                                                          # by checking clean background image and grid/mesh reference image. So, exclude the center
                                                          # without reaching the first grid pattern frequency peaks.

        self.max_peaks_to_find = 20                       # Alarm threshold: if more than this many peaks are detected, raise a warning.
                                                          # Grid pattern appears too noisy or irregular. The mesh/grid should be regular, periodic
                                                          # and is imaged to be as noise-free as possible.

        # Adaptive window sizing
        # ----------------------
        self.nyquist_sampling_factor = 2.0        # Nyquist criterion for adequate sampling
        self.gaussian_sigma_multiplier = 3.0      # 3-sigma rule for Gaussian analysis window
        self.statistical_confidence_level = 0.95  # Statistical confidence for width estimation
        self.half_max_threshold = 0.5             # Fraction of peak max for FWHM calculation
        self.angular_resolution_degrees = 22.5    # Angular resolution for isotropic sampling (22.5 deg = π/8 rad))
        self.robust_estimator_percentile = 0.8    # Percentile for statistical estimation

        # Peak Detection initial guesses
        # ------------------------------
        self.peak_threshold = 0.1      
        self.peak_min_distance = 10            
        self.gaussian_sigma = 2.0             
        
        # Parameter Optimization Bounds
        # -----------------------------
        self.peak_threshold_bounds = (0.01, 0.5)
        self.peak_min_distance_bounds = (1, 50)       
        self.gaussian_sigma_bounds = (0.5, 10.0)     
        
        # Filter Creation
        # ---------------
        self.optimal_filter_radius = None 
        self.detected_grid_frequencies = None
        self.final_quality_score = None 




# ============= THE GRID REMOVAL CLASS =======================================================

class GridRemoval:
    def __init__(self, dark_field_path=None, reference_path=None, flat_field_path=None):
        self.params = ProcessingParams()
        
        # BACKUP: If paths are not provided, use file dialogs to select them
        if dark_field_path is None or reference_path is None or flat_field_path is None:
            root = tk.Tk()
            root.withdraw()
            
            # SELECT dark-field image with grid artifact
            if dark_field_path is None:
                dark_field_path = filedialog.askopenfilename(
                    title="Select the object image with the grid artifact", 
                    filetypes=[("TIFF files", "*.tif *.tiff"), ("All Files", "*.*")]
                )
                if not dark_field_path:
                    raise Exception("User canceled the image selection.")
            
            # SELECT reference image with only grid artifact
            if reference_path is None:
                reference_path = filedialog.askopenfilename(
                    title="Select reference image with only the grid artifact", 
                    filetypes=[("TIFF files", "*.tif *.tiff"), ("All Files", "*.*")]
                )
                if not reference_path:
                    raise Exception("User canceled the reference image selection.")
            
            # SELECT clean flat field image
            if flat_field_path is None:
                flat_field_path = filedialog.askopenfilename(
                    title="Select clean flat field image (without grid)", 
                    filetypes=[("TIFF files", "*.tif *.tiff"), ("All Files", "*.*")]
                )
                if not flat_field_path:
                    raise Exception("User canceled the flat field image selection.")
            
            root.destroy()
        
        self.dark_field_path = dark_field_path
        self.reference_path = reference_path
        self.flat_field_path = flat_field_path
        
        self._load_images()
        self.auto_set_peak_param_bounds()


    def auto_set_peak_param_bounds(self):
        """
        Automatically set peak detection parameter bounds based on image data statistics.

        Args:
            None
                It uses:
                self.reference_working - the preprocessed reference image (with grid pattern).
                self.flat_field_working - the preprocessed flat field image (without grid pattern).
                self.dark_field_working - the preprocessed dark field image (with grid pattern).

        Returns:
            None
                It updates:
                self.params.peak_threshold_bounds - tuple of (lower_bound, upper_bound) for peak threshold.
                self.params.peak_min_distance_bounds - tuple of (min_distance, max_distance) for peak distance.
                self.params.gaussian_sigma_bounds - tuple of (min_sigma, max_sigma) for Gaussian smoothing.

        """
        try:
            # COMPUTE FFT difference
            diff_mag, _, _ = self._compute_fft_difference(self.reference_working, self.flat_field_working)

            old_threshold = self.params.peak_threshold
            old_min_distance = self.params.peak_min_distance
            old_sigma = self.params.gaussian_sigma
            self.params.peak_threshold = 0.05
            self.params.peak_min_distance = 5
            self.params.gaussian_sigma = 2.0

            peak_coords, peak_values = self._detect_peaks_with_exclusion(diff_mag)

            # PEAK THRESHOLD BOUNDS: based on min/max peak value
            if len(peak_values) > 0:
                max_peak = float(np.max(peak_values))
                min_peak = float(np.min(peak_values))
                # Lower bound: 1% of max, upper bound: 80% of max
                lower = max(0.01, 0.01 * max_peak)
                upper = max(0.1, 0.8 * max_peak)
                self.params.peak_threshold_bounds = (lower, upper)
            else:
                self.params.peak_threshold_bounds = (0.01, 0.5)

            # PEAK MIN DISTANCE BOUNDS: based on min/max distance between peaks
            if len(peak_coords) > 1:
                from scipy.spatial.distance import pdist
                dists = pdist(peak_coords)
                min_dist = max(1, np.percentile(dists, 10))
                max_dist = max(min_dist+1, np.percentile(dists, 90))
                self.params.peak_min_distance_bounds = (int(min_dist), int(max_dist))
            else:
                self.params.peak_min_distance_bounds = (1, 50)

            # GAUSSIAN SIGMA BOUNDS: based on FWHM of peaks
            if len(peak_coords) > 0:
                fwhms = []
                for y, x in peak_coords[:min(5, len(peak_coords))]:
                    peak_intensity = diff_mag[y, x]
                    half_max = peak_intensity * self.params.half_max_threshold
                    # SETUP the 4 directions (up, down, left, right)
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        for sign in [1, -1]:
                            step = 1
                            while True:
                                yy = y + sign*step*dy
                                xx = x + sign*step*dx
                                if 0 <= yy < diff_mag.shape[0] and 0 <= xx < diff_mag.shape[1]:
                                    if diff_mag[int(yy), int(xx)] < half_max:
                                        fwhms.append(step)
                                        break
                                    step += 1
                                    if step > 20:
                                        break
                                else:
                                    break
                if len(fwhms) > 0:
                    min_sigma = max(0.5, np.percentile(fwhms, 10)/2.355)
                    max_sigma = max(min_sigma+0.1, np.percentile(fwhms, 90)/2.355)
                    self.params.gaussian_sigma_bounds = (float(min_sigma), float(max_sigma))
                else:
                    self.params.gaussian_sigma_bounds = (0.5, 10.0)
            else:
                self.params.gaussian_sigma_bounds = (0.5, 10.0)

            self.params.peak_threshold = old_threshold
            self.params.peak_min_distance = old_min_distance
            self.params.gaussian_sigma = old_sigma

            print(f"Auto-set peak_threshold_bounds: {self.params.peak_threshold_bounds}")
            print(f"Auto-set peak_min_distance_bounds: {self.params.peak_min_distance_bounds}")
            print(f"Auto-set gaussian_sigma_bounds: {self.params.gaussian_sigma_bounds}")
        except Exception as e:
            print(f"Warning: Could not auto-set peak parameter bounds: {e}")
        

    def _load_images(self):
        """
        Load all three images and prepare them for processing.

        Args:
            None 
                It uses:
                self.dark_field_path - path to the dark-field image with grid artifact.
                self.reference_path - path to the reference image with only grid artifact.
                self.flat_field_path - path to the clean flat field image (without grid).


        Returns:
            None
                It updates:
                self.dark_field_image - the loaded dark-field image as a numpy array.
                self.reference_image - the loaded reference image as a numpy array.
                self.flat_field_image - the loaded flat field image as a numpy array.

        """
        try:
            # LOAD dark-field image
            dark_img_pil = Image.open(self.dark_field_path)
            self.dark_field_image = np.array(dark_img_pil)
            self.orig_mode = dark_img_pil.mode
            self.orig_dtype = self.dark_field_image.dtype
            
            # LOAD grid reference image
            ref_img_pil = Image.open(self.reference_path)
            self.reference_image = np.array(ref_img_pil)
            
            # LOAD flat field image
            flat_img_pil = Image.open(self.flat_field_path)
            self.flat_field_image = np.array(flat_img_pil)
            
            # CHECK formats 
            if len(self.dark_field_image.shape) == 3:
                self.working_image = cv2.cvtColor(self.dark_field_image, cv2.COLOR_RGB2GRAY)
            else:
                self.working_image = self.dark_field_image.copy()
            if len(self.reference_image.shape) == 3:
                self.reference_working = cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2GRAY)
            else:
                self.reference_working = self.reference_image.copy()
            if len(self.flat_field_image.shape) == 3:
                self.flat_field_working = cv2.cvtColor(self.flat_field_image, cv2.COLOR_RGB2GRAY)
            else:
                self.flat_field_working = self.flat_field_image.copy()
                
            # NORMALIZE all images to [0, 1] for processing
            self.working_image = self.working_image.astype(np.float64)
            self.img_min, self.img_max = self.working_image.min(), self.working_image.max()
            if self.img_max > self.img_min:
                self.working_image = (self.working_image - self.img_min) / (self.img_max - self.img_min)
            
            self.reference_working = self.reference_working.astype(np.float64)
            ref_min, ref_max = self.reference_working.min(), self.reference_working.max()
            if ref_max > ref_min:
                self.reference_working = (self.reference_working - ref_min) / (ref_max - ref_min)
            
            self.flat_field_working = self.flat_field_working.astype(np.float64)
            flat_min, flat_max = self.flat_field_working.min(), self.flat_field_working.max()
            if flat_max > flat_min:
                self.flat_field_working = (self.flat_field_working - flat_min) / (flat_max - flat_min)
            
            print(f"Loaded dark-field image: {self.working_image.shape}, dtype: {self.orig_dtype}")
            print(f"Loaded reference grid image: {self.reference_working.shape}")
            print(f"Loaded flat field image: {self.flat_field_working.shape}")
            
        except Exception as e:
            raise Exception(f"Error loading images: {str(e)}")
    

    def _compute_fft_difference(self, reference_img, clean_img):
        """
        Compute the FFT difference between reference (with grid) and clean images.
        This helps the algorithm to notice mainly the grid pattern.
        
        Args:
            reference_img - The reference image containing the grid pattern (2D numpy array).
            clean_img - The clean flat field image without the grid pattern (2D numpy array).

        Returns:
            diff_mag - The normalized difference of the FFT magnitudes (highlights grid frequencies).
            ref_fft - The FFT of the reference image.
            clean_fft - The FFT of the clean image.

        """
        # COMPUTE FFTs
        ref_fft = fftshift(fft2(reference_img))
        clean_fft = fftshift(fft2(clean_img))
        
        # COMPUTE magnitude difference (grid components should be prominent)
        ref_mag = np.abs(ref_fft)
        clean_mag = np.abs(clean_fft)
        
        # NORMALIZE magnitudes
        ref_mag_norm = ref_mag / (np.max(ref_mag) + 1e-10)
        clean_mag_norm = clean_mag / (np.max(clean_mag) + 1e-10)
        
        # HIGHLIGHT grid frequencies
        diff_mag = ref_mag_norm - clean_mag_norm
        
        # APPLY Gaussian to enhance peak detection
        diff_mag = gaussian_filter(diff_mag, sigma=self.params.gaussian_sigma)
        
        return diff_mag, ref_fft, clean_fft
    

    def _detect_peaks_with_exclusion(self, magnitude_spectrum):
        """
        Detect peaks in the magnitude spectrum while excluding DC component.
        
        Args:
            magnitude_spectrum -  Magnitude spectrum (FFT magnitude, 2D numpy array).
            
        Returns:
            peak_coords - Coordinates of detected peaks (2D numpy array).
            peak_values - Values of detected peaks (1D numpy array).

        """
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # CREATE MASK to exclude DC component
        y, x = np.ogrid[:h, :w]
        dc_mask = (x - center_x)**2 + (y - center_y)**2 > self.params.exclude_dc_radius**2
        
        # APPLY mask
        masked_spectrum = magnitude_spectrum.copy()
        masked_spectrum[~dc_mask] = 0
        
        # FIND local maxima using scipy's maximum_filter
        # CREATE a local maxima mask
        local_maxima = (masked_spectrum == maximum_filter(masked_spectrum, size=self.params.peak_min_distance))
        
        # APPLY threshold
        threshold = self.params.peak_threshold * np.max(masked_spectrum)
        peak_mask = local_maxima & (masked_spectrum > threshold)
        
        # GET coordinates and values
        coords = np.where(peak_mask)
        
        if len(coords[0]) > 0:
            peak_coords = np.column_stack((coords[0], coords[1]))
            peak_values = masked_spectrum[coords]
            # SORT peaks by value (descending)
            sort_idx = np.argsort(peak_values)[::-1]
            peak_coords = peak_coords[sort_idx]
            peak_values = peak_values[sort_idx]
            
            return peak_coords, peak_values
        else:
            return np.array([]), np.array([])
    

    def _objective_function(self, params_vec, reference_img, clean_img):
        """
        Objective function for L-BFGS-B optimization.
        Measures the quality of peak detection parameters.

        This function helps to find automatically the best settings for detecting the grid pattern in the image.
        It works by:
        1. Trying a set of detection settings (threshold, distance, smoothing).
        2. Detecting peaks (possible grid lines) in the frequency domain using these settings.
        3. Scoring how well these settings work:
            - If no or too many peaks are found, the score is low (bad).
            - If a reasonable number of strong, well-distributed peaks are found, the score is high (good).
        4. The optimizer repeats this process with different settings, always trying to improve the score.
        5. The best settings are used for the rest of the grid removal process.

        This function tells the optimizer how "good" a set of peak detection settings is, so it can automatically 
        find the most effective way to detect and remove the grid pattern from your images.

        Args:
            params_vec - Vector of parameters [threshold, min_distance, sigma]
            reference_img - Reference image with grid (2D numpy array)
            clean_img - Clean flat field image (2D numpy array)

        Returns:
            -quality - Negative quality score for minimization

        """
        threshold, min_distance, sigma = params_vec
        
        # STORE old parameters to restore later
        old_threshold = self.params.peak_threshold
        old_min_distance = self.params.peak_min_distance
        old_sigma = self.params.gaussian_sigma
        
        self.params.peak_threshold = threshold
        self.params.peak_min_distance = max(1, int(min_distance))
        self.params.gaussian_sigma = sigma
        
        try:
            # COMPUTE FFT difference
            diff_mag, _, _ = self._compute_fft_difference(reference_img, clean_img)
            
            # DETECT peaks
            peak_coords, peak_values = self._detect_peaks_with_exclusion(diff_mag)
            
            if len(peak_coords) == 0:
                quality = -1000  # Overly excessive penalty for no peaks detected, to notify the optimizer that this is a bad setting.
            else:
                h, w = diff_mag.shape
                center_y, center_x = h // 2, w // 2
                
                # CONVERT to normalized frequencies
                freq_coords = []
                for y, x in peak_coords:
                    freq_y = (y - center_y) / h
                    freq_x = (x - center_x) / w
                    freq_mag = np.sqrt(freq_y**2 + freq_x**2)
                    freq_coords.append([freq_y, freq_x, freq_mag])
                freq_coords = np.array(freq_coords)
                
                # FILTER peaks in expected frequency range
                valid_mask = (freq_coords[:, 2] >= self.params.expected_grid_frequency_range[0]) & \
                           (freq_coords[:, 2] <= self.params.expected_grid_frequency_range[1])
                valid_peaks = np.sum(valid_mask)
                valid_peak_values = peak_values[valid_mask] if valid_peaks > 0 else []
                
                # =================================================================================
                # SCORING
                # 1. Peak strength Z-score: how much stronger are peaks than the background?
                # 2. Peak contrast Z-score: how much do peak strengths vary compared to background?
                # 3. Peak count penalty if too few or too many peaks are found.
                # =================================================================================
                # COMPUTE background statistics
                all_values = peak_values if len(peak_values) > 0 else np.array([0])
                bg_mask = ~valid_mask if len(valid_mask) == len(peak_values) else np.array([False]*len(peak_values))
                bg_values = peak_values[bg_mask] if np.any(bg_mask) else all_values
                bg_mean = np.mean(bg_values) if len(bg_values) > 0 else 1e-6
                bg_std = np.std(bg_values) if len(bg_values) > 1 else 1e-6

                # PEAK STRENGTH Z-score (mean of valid peaks relative to background)
                mean_strength = np.mean(valid_peak_values) if len(valid_peak_values) > 0 else 0
                z_strength = (mean_strength - bg_mean) / (bg_std + 1e-6)

                # PEAK CONTRAST Z-score (std of valid peaks relative to background)
                peak_contrast = np.std(valid_peak_values) if len(valid_peak_values) > 1 else 0
                z_contrast = (peak_contrast - bg_std) / (bg_std + 1e-6)

                # PEAK COUNT PENALTY (using parameter bounds)
                min_peaks = getattr(self.params, 'min_peaks_allowed', 2) # Grid patterns must have at least 2 peaks for it to be a grid.
                max_peaks = getattr(self.params, 'max_peaks_to_find', 20)
                # If within bounds, no penalty. If outside, penalty increases with distance.
                if valid_peaks < min_peaks:
                    count_penalty = -((min_peaks - valid_peaks) / min_peaks)
                elif valid_peaks > max_peaks:
                    count_penalty = -((valid_peaks - max_peaks) / max_peaks)
                else:
                    count_penalty = 0.0

                # SUM ALL as quality score
                quality = z_strength + z_contrast + count_penalty
            
        except Exception as e:
            quality = -1000  # Overly excessive penalty for no peaks detected, to notify the optimizer that this is a bad setting.
            print(f"Error in objective function: {e}")
        finally:
            # RESTORE original parameters
            self.params.peak_threshold = old_threshold
            self.params.peak_min_distance = old_min_distance
            self.params.gaussian_sigma = old_sigma
        
        return -quality
    

    def optimize_peak_detection_parameters(self):
        """
        Use L-BFGS-B to optimize parameters for peak detection in grid reference image.
        
        Args:
            None 
                It uses:
                self.reference_working - the preprocessed reference image (with grid pattern).
                self.flat_field_working - the preprocessed flat field image (without grid pattern).
                self.params - the parameter object containing all optimization and detection settings.

        Returns:
            result, grid_peaks - Optimized parameters and detected grid peaks

        """
        print("Optimizing peak detection parameters using L-BFGS-B...")
        
        # INITIALIZE guess
        x0 = np.array([
            self.params.peak_threshold,
            self.params.peak_min_distance,
            self.params.gaussian_sigma
        ])
        
        # BOUNDS from ProcessingParams
        bounds = [
            self.params.peak_threshold_bounds,      # peak_threshold
            self.params.peak_min_distance_bounds,   # peak_min_distance
            self.params.gaussian_sigma_bounds       # gaussian_sigma
        ]
        
        # RUN optimization
        result = minimize(
            self._objective_function,
            x0,
            args=(self.reference_working, self.flat_field_working),
            method=self.params.optimization_method,
            bounds=bounds,
            options={
                'maxiter': self.params.max_iterations,
                'ftol': self.params.ftol,
                'gtol': self.params.gtol,
                'disp': True
            }
        )
        
        if result.success:
            # UPDATE parameters with optimized values
            self.params.peak_threshold = result.x[0]
            self.params.peak_min_distance = int(result.x[1])
            self.params.gaussian_sigma = result.x[2]
            
            print(f"Optimization successful!")
            print(f"Optimized parameters:")
            print(f"  Peak threshold: {self.params.peak_threshold:.4f}")
            print(f"  Min distance: {self.params.peak_min_distance}")
            print(f"  Gaussian sigma: {self.params.gaussian_sigma:.4f}")
            
            # DETECT peaks with optimized parameters
            grid_peaks = self.detect_grid_peaks()
            
            return result, grid_peaks
        else:
            print(f"Optimization failed: {result.message}")
            return result, None
    

    def detect_grid_peaks(self):
        """
        Detect grid peaks in the reference image using the flat field as clean reference.
       
         Args:
            None
                It uses:
                self.reference_working - the preprocessed reference image (with grid pattern).
                self.flat_field_working - the preprocessed flat field image (without grid pattern).

        Returns:
            grid_peaks:
                peak_coords - Coordinates of detected peaks (2D numpy array),
                peak_values - Values of detected peaks (1D numpy array),
                frequencies - Frequencies of detected peaks (3D numpy array),
                magnitude_spectrum - Magnitude spectrum of the FFT difference (2D numpy array),
                reference_fft - FFT of the reference image (2D numpy array),
                clean_fft - FFT of the clean flat field image (2D numpy array)

        """

        print("Detecting grid peaks in Fourier space...")
        
        # COMPUTE FFT difference between reference and flat field
        diff_mag, ref_fft, clean_fft = self._compute_fft_difference(
            self.reference_working, self.flat_field_working
        )
        
        # DETECT peaks
        peak_coords, peak_values = self._detect_peaks_with_exclusion(diff_mag)
        
        if len(peak_coords) == 0:
            print("No grid peaks detected!")
            return None

        # ALARM: warn if too many peaks are detected (possible noise or irregular grid)
        if len(peak_coords) > self.params.max_peaks_to_find:
            print(f"Warning: Detected {len(peak_coords)} peaks, which exceeds the alarm threshold ({self.params.max_peaks_to_find}).")
            print("This may indicate excessive noise, artifacts, or an irregular grid pattern in the image.")
        
        # CONVERT peak coordinates to frequencies
        h, w = diff_mag.shape
        center_y, center_x = h // 2, w // 2
        
        grid_peaks = {
            'peak_coords': peak_coords,
            'peak_values': peak_values,
            'frequencies': [],
            'magnitude_spectrum': diff_mag,
            'reference_fft': ref_fft,
            'clean_fft': clean_fft
        }
        
        for y, x in peak_coords:
            freq_y = (y - center_y) / h
            freq_x = (x - center_x) / w
            freq_mag = np.sqrt(freq_y**2 + freq_x**2)
            grid_peaks['frequencies'].append([freq_y, freq_x, freq_mag])
        
        grid_peaks['frequencies'] = np.array(grid_peaks['frequencies'])
        
        # FILTER peaks
        valid_mask = (grid_peaks['frequencies'][:, 2] >= self.params.expected_grid_frequency_range[0]) & \
                    (grid_peaks['frequencies'][:, 2] <= self.params.expected_grid_frequency_range[1])
        
        valid_peaks = np.sum(valid_mask)
        
        print(f"Found {len(peak_coords)} total peaks, {valid_peaks} in expected frequency range")
        
        if valid_peaks > 0:
            print("Valid grid frequencies (fy, fx, magnitude):")
            for i, (freq, coord, value) in enumerate(zip(
                grid_peaks['frequencies'][valid_mask],
                peak_coords[valid_mask],
                peak_values[valid_mask]
            )):
                print(f"  Peak {i+1}: freq=({freq[0]:.4f}, {freq[1]:.4f}), "
                      f"mag={freq[2]:.4f}, strength={value:.4f}")
        
        return grid_peaks
    

    def _estimate_peak_fwhm_statistics(self, magnitude_spectrum, peak_coords):
        """
        Estimate Full Width at Half Maximum (FWHM) of the detected grid peaks in the FFT magnitude spectrum.
        
        Args:
            magnitude_spectrum - 2D magnitude spectrum of the FFT difference 
            peak_coords - Coordinates of detected peaks (2D numpy array)
            
        Returns:
            estimated_radius - Estimated radius (in pixels) for the analysis window around each peak.

            
        """
        if len(peak_coords) == 0:
            # FALLBACK: minimum window based on Nyquist criterion
            return max(3, int(self.params.nyquist_sampling_factor))
        
        fwhm_measurements = []
        
        # PROCESS the detected peaks for FWHM estimation
        sample_peaks = peak_coords
        
        for peak_y, peak_x in sample_peaks:
            peak_intensity = magnitude_spectrum[peak_y, peak_x]
            half_max_threshold = peak_intensity * self.params.half_max_threshold

            # INITIALIZE directional widths list
            directional_widths = []

            # CALCULATE directional widths. Sample in multiple directions around the peak
            angular_resolution_rad = np.deg2rad(self.params.angular_resolution_degrees)
            n_directions = int(np.pi / angular_resolution_rad)
            angular_step = np.pi / n_directions
            
            for angle_idx in range(n_directions):
                angle = angle_idx * angular_step
                direction_vector = np.array([np.cos(angle), np.sin(angle)])

                # SAMPLE along direction until half-maximum
                width_estimate = self._sample_directional_width(
                    magnitude_spectrum, peak_y, peak_x, 
                    direction_vector, half_max_threshold
                )
                # STORE width estimate if valid
                if width_estimate > 0:
                    directional_widths.append(width_estimate)
            
            # CHECK if we have enough directional widths to estimate FWHM
            if len(directional_widths) >= 2:
                robust_width = np.percentile(directional_widths, 
                                           self.params.robust_estimator_percentile * 100)
                fwhm_measurements.append(robust_width)
        
        # ESTIMATE the analysis radius based on FWHM measurements
        # If we have FWHM measurements, then proceed
        if len(fwhm_measurements) > 0:
            # MEAN and STD of FWHM measurements
            mean_fwhm = np.mean(fwhm_measurements)
            std_fwhm = np.std(fwhm_measurements) if len(fwhm_measurements) > 1 else 0
            
            # CONFIDENCE INTERVAL based on statistical confidence level
            z_score = 1.96 if self.params.statistical_confidence_level == 0.95 else 2.58  # 95% or 99%
            confidence_margin = z_score * std_fwhm / np.sqrt(len(fwhm_measurements))
            
            # ADAPTIVE analysis radius based on FWHM and confidence margin
            analysis_radius = (mean_fwhm + confidence_margin) * self.params.gaussian_sigma_multiplier
            
            # DATA-DRIVEN max radius. Use half the minimum distance between detected peaks
            estimated_radius = int(np.ceil(analysis_radius))
            min_radius = max(3, int(self.params.nyquist_sampling_factor))
            if len(peak_coords) > 1:
                from scipy.spatial.distance import pdist
                dists = pdist(peak_coords)
                min_peak_dist = np.min(dists)
                max_radius = max(min_radius, int(min_peak_dist / 2))
            else:
                max_radius = max(min_radius, int(min(magnitude_spectrum.shape) / 4))  # FALLBACK 1/4 of smallest dimension
            return np.clip(estimated_radius, min_radius, max_radius)
        else:
            # FALLBACK based on Nyquist sampling
            return max(3, int(self.params.nyquist_sampling_factor))
    

    def _sample_directional_width(self, spectrum, center_y, center_x, direction, threshold):
        """
        Sample the width of a detected peak in the FFT magnitude spectrum by moving outward from the peak center along a specified direction.
        Continue sampling until the intensity drops below a threshold.
        
        Args:
            spectrum - 2D magnitude spectrum of the FFT difference
            center_y, center_x - Peak center coordinates
            direction - Unit direction vector [dy, dx]
            threshold - Intensity threshold for width estimation

        Returns:
            step - Estimated width in pixels along the direction until the threshold is crossed.

        """
        # USE adaptive analysis radius, else FALLBACK to Nyquist-based minimum
        if hasattr(self, '_get_adaptive_analysis_radius'):
            max_search_radius = max(3, int(self.params.nyquist_sampling_factor))
        else:
            print("Warning: Adaptive analysis radius not available, using Nyquist-based minimum search radius.")
            max_search_radius = max(3, int(min(spectrum.shape) / 4))
        dy, dx = direction

        for step in range(1, max_search_radius + 1):
            # POSITIONS
            y_pos = center_y + step * dy
            x_pos = center_x + step * dx

            # CHECK bounds
            if (0 <= y_pos < spectrum.shape[0] - 1 and 
                0 <= x_pos < spectrum.shape[1] - 1):

                # INTERPOLATE
                y_floor, x_floor = int(y_pos), int(x_pos)
                y_frac, x_frac = y_pos - y_floor, x_pos - x_floor
                intensity = (spectrum[y_floor, x_floor] * (1 - y_frac) * (1 - x_frac) +
                           spectrum[y_floor + 1, x_floor] * y_frac * (1 - x_frac) +
                           spectrum[y_floor, x_floor + 1] * (1 - y_frac) * x_frac +
                           spectrum[y_floor + 1, x_floor + 1] * y_frac * x_frac)

                if intensity < threshold:
                    return step
            else:
                break

        return 0
    
    
    def _get_adaptive_analysis_radius(self, image_shape, peak_coords=None, magnitude_spectrum=None):
        """
        Estimate the best analysis radius (window size) for grid peak analysis in the frequency domain.

        - If we have detected grid peaks, we measure their width and use statistics to set the window size.
        - If we don't have enough data, we use a safe minimum size based on the Nyquist sampling rule.
        - The window size is always kept within bounds to avoid errors near image edges.

        Args:
            image_shape - Shape of the image (height, width)
            peak_coords - Coordinates of detected peaks (2D numpy array)
            magnitude_spectrum - FFT magnitude spectrum (2D numpy array)

        Returns:
            final_radius - Estimated analysis radius (in pixels) for grid peak analysis.

        """
        # NYQUIST minimum radius
        nyquist_min_radius = max(3, int(self.params.nyquist_sampling_factor))

        # DATA-DRIVEN maximum radius: half the minimum distance between detected peaks
        if peak_coords is not None and len(peak_coords) > 1:
            from scipy.spatial.distance import pdist
            dists = pdist(peak_coords)
            min_peak_dist = np.min(dists)
            max_radius = max(nyquist_min_radius, int(min_peak_dist / 2))
        else:
            max_radius = max(nyquist_min_radius, int(min(image_shape) / 4))  # FALLBACK
        
        # CHECK if we have peak coordinates and magnitude spectrum
        if peak_coords is not None and magnitude_spectrum is not None and len(peak_coords) > 0:
            # USE FWHM estimation with statistical confidence
            statistical_radius = self._estimate_peak_fwhm_statistics(magnitude_spectrum, peak_coords)
            # ENSURE radius is within bounds
            final_radius = np.clip(statistical_radius, nyquist_min_radius, max_radius)
        else:
            # FALLBACK to Nyquist-based estimate with 3-sigma analysis window
            fallback_radius = int(nyquist_min_radius * self.params.gaussian_sigma_multiplier)
            final_radius = np.clip(fallback_radius, nyquist_min_radius, max_radius)
        
        return final_radius
        
    
    
    def process_grid_detection(self, visualize=True, optimize_params=True):
        """
        Automatically detects the grid pattern in the image using Fourier space analysis.

        Args:
            optimize_params - Whether to automatically optimize detection settings.

        Returns:
            grid_peaks - Dictionary containing detected grid peak information, or None if detection failed.
        """
        try:
            if optimize_params:
                print("Starting automatic grid detection with parameter optimization...")
                optimization_result, grid_peaks = self.optimize_peak_detection_parameters()
                if grid_peaks is None:
                    print("Parameter optimization failed, trying with default parameters...")
                    grid_peaks = self.detect_grid_peaks()
            else:
                print("Starting grid detection with current parameters...")
                grid_peaks = self.detect_grid_peaks()
            return grid_peaks
        except Exception as e:
            print(f"Error in grid detection process: {str(e)}")
            return None
    

    def create_grid_filter(self, grid_peaks, filter_radius=None):
        """
        Creates a filter that removes the grid pattern from the image in the Fourier space.
        
        Args:
            grid_peaks - Dictionary containing detected grid peak information
            filter_radius - Radius of notch filters around each peak (if None, uses default)

        Returns:
            filter_mask - 2D numpy array representing the notch filter in Fourier space

        """
        if grid_peaks is None or len(grid_peaks['peak_coords']) == 0:
            print("No grid peaks available for filter creation!")
            return None
        
        # FALLBACK: use default filter radius if not specified
        if filter_radius is None:
            filter_radius = self.params.default_filter_radius
        
        h, w = grid_peaks['magnitude_spectrum'].shape
        center_y, center_x = h // 2, w // 2

        # INITIALIZE filter mask with ones
        filter_mask = np.ones((h, w), dtype=np.float64)
        
        # CREATE coordinate grids
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # FILTER peaks
        valid_mask = (grid_peaks['frequencies'][:, 2] >= self.params.expected_grid_frequency_range[0]) & \
                    (grid_peaks['frequencies'][:, 2] <= self.params.expected_grid_frequency_range[1])
        
        valid_peaks = grid_peaks['peak_coords'][valid_mask]
        
        print(f"Creating notch filters for {len(valid_peaks)} valid grid peaks...")
        
        for i, (peak_y, peak_x) in enumerate(valid_peaks):
            # CREATE circular notch filter around this peak
            distance = np.sqrt((x_coords - peak_x)**2 + (y_coords - peak_y)**2)
            
            # SMOOTH notch filter
            notch = 1.0 - np.exp(-(distance**2) / (2 * filter_radius**2))
            
            # APPLY main filter
            filter_mask *= notch
            
            # ALSO FILTER the symmetric peaks due to FFT symmetry
            sym_y = 2 * center_y - peak_y
            sym_x = 2 * center_x - peak_x
            
            if 0 <= sym_y < h and 0 <= sym_x < w:
                distance_sym = np.sqrt((x_coords - sym_x)**2 + (y_coords - sym_y)**2)
                notch_sym = 1.0 - np.exp(-(distance_sym**2) / (2 * filter_radius**2))
                filter_mask *= notch_sym
        
        print(f"Filter created with {len(valid_peaks)} notch filters (radius={filter_radius})")
        return filter_mask
    
    
    def remove_grid_from_image(self, image, grid_peaks, filter_radius=None):
        """
        Remove grid artifacts from an image using the detected grid peaks.
        
        Args:
            image - Input image to degrid (2D numpy array)
            grid_peaks - Dictionary containing detected grid peak information
            filter_radius - Radius of notch filters around each peak (if None, uses default)
            
        Returns:
            Degridded_image - Image with grid artifacts removed (2D numpy array)

        """
        if grid_peaks is None:
            print("No grid peaks available for grid removal!")
            return image
        
        # FALLBACK: use default filter radius if not specified
        if filter_radius is None:
            filter_radius = self.params.default_filter_radius
        
        print("Removing grid artifacts from image...")
        
        # FFT of the input image
        img_fft = fftshift(fft2(image))
        
        # CREATE grid filter
        grid_filter = self.create_grid_filter(grid_peaks, filter_radius)
        if grid_filter is None:
            return image
        
        # APPLY filter in frequency domain
        filtered_fft = img_fft * grid_filter
        
        # CONVERT back to spatial domain
        degridded_image = np.real(ifft2(ifftshift(filtered_fft)))
        
        # ENSURE the output is in the same range as input
        degridded_image = np.clip(degridded_image, 0, 1)
        
        print("Grid removal completed.")
        return degridded_image
    
    
    def adaptive_grid_removal(self, image, grid_peaks):
        """
        Adaptive grid removal using MSE-based optimization for filter parameter selection.
        
        Uses Mean Squared Error (MSE) to quantify grid suppression at detected frequencies.
        Tests multiple filter radii and selects the one with the best MSE suppression score.
        
        Args:
            image - Input image to degrid (2D numpy array)
            grid_peaks - Dictionary containing detected grid peak information

        Returns:
            best_image - Image with grid artifacts removed (2D numpy array) 
            best_radius - Optimal filter radius used for grid removal

        """
        if grid_peaks is None:
            return image, self.params.base_filter_radius
        
        print("Starting adaptive grid removal (MSE-based optimization)...")
        
        # INITIALIZE image and quality score
        best_image = image
        best_radius = self.params.base_filter_radius
        best_quality = -1  # Start with negative to ensure positive score is better
        
        # TRY different filter radii
        for radius in range(self.params.base_filter_radius, self.params.max_filter_radius + 1):
            print(f"Testing filter radius: {radius}")
            
            # REMOVE grid with current radius
            degridded = self.remove_grid_from_image(image, grid_peaks, radius)
            
            # EVALUATE quality using MSE 
            quality = self._evaluate_degridding_quality(image, degridded, grid_peaks)
            
            print(f"  MSE grid suppression score: {quality:.6f}")
            
            # UPDATE if we get better grid suppression
            if quality > best_quality:
                best_quality = quality
                best_image = degridded
                best_radius = radius
                print(f"  -> New best radius: {radius} (MSE score: {quality:.6f})")

            # CONTINUE testing
            if quality > 0.99:
                print(f"Near-optimal grid suppression achieved at radius {radius}")
                break
        
        print(f"Optimal filter radius: {best_radius} (MSE suppression: {best_quality:.6f})")
        return best_image, best_radius
    
    
    def _evaluate_degridding_quality(self, original, degridded, grid_peaks):
        """
        Evaluate the quality of grid removal using Mean Squared Error (MSE) analysis.
        
        Measures FFT magnitude reduction at grid frequencies in the Fourier space.
        Higher quality score indicates better grid suppression.
        
        Args:
            original - Original image with the grid artifact
            degridded - Degridded image
            grid_peaks - Dictionary containing detected grid peak information

        Returns:
            final_quality - Quality score of the grid removal (float)

        """

        try:
            # FFTs for frequency space analysis
            orig_fft = fftshift(fft2(original))
            degrid_fft = fftshift(fft2(degridded))
            orig_mag = np.abs(orig_fft)
            degrid_mag = np.abs(degrid_fft)
            
            # GET grid peaks
            valid_mask = (grid_peaks['frequencies'][:, 2] >= self.params.expected_grid_frequency_range[0]) & \
                        (grid_peaks['frequencies'][:, 2] <= self.params.expected_grid_frequency_range[1])
            
            valid_peaks = grid_peaks['peak_coords'][valid_mask]
            
            if len(valid_peaks) == 0:
                return 0.0
            
            # CALCULATE adaptive analysis radius
            adaptive_radius = self._get_adaptive_analysis_radius(
                orig_mag.shape, valid_peaks, grid_peaks['magnitude_spectrum']
            )
            
            print(f"Using statistically-optimized analysis radius: {adaptive_radius} pixels")
            
            grid_suppression_scores = []
            
            for peak_y, peak_x in valid_peaks:
                # DEFINE analysis region around grid peak
                radius = adaptive_radius
                y1, y2 = max(0, peak_y-radius), min(orig_mag.shape[0], peak_y+radius+1)
                x1, x2 = max(0, peak_x-radius), min(orig_mag.shape[1], peak_x+radius+1)
                
                # EXTRACT regions at grid frequency locations
                orig_region = orig_mag[y1:y2, x1:x2]
                degrid_region = degrid_mag[y1:y2, x1:x2]

                # CALCULATE reduction at grid frequencies
                if orig_region.size > 0:
                    orig_energy = np.mean(orig_region**2)
                    degrid_energy = np.mean(degrid_region**2)
                    
                    # GRID suppression ratio
                    if orig_energy > 1e-10:  # Avoid division by zero
                        suppression = 1.0 - (degrid_energy / orig_energy)
                        suppression = max(0.0, min(1.0, suppression))
                        grid_suppression_scores.append(suppression)

            # MEAN grid suppression scores
            if len(grid_suppression_scores) > 0:
                grid_suppression = np.mean(grid_suppression_scores)
            else:
                grid_suppression = 0.0
            
            # APPLY confidence weighting based on number of peaks analyzed
            confidence_factor = min(1.0, len(grid_suppression_scores) / self.params.confidence_peak_threshold)
            final_quality = grid_suppression * confidence_factor
            
            return final_quality
            
        except Exception as e:
            print(f"Error evaluating quality: {e}")
            return 0.0
    

    def visualize_grid_removal_results(self, original, degridded, grid_peaks, optimal_radius=None):
        """
        Visualize i.e. show plots of the grid removal.
        
        Args:
            original - Original image with grid
            degridded - Degridded image
            grid_peaks - Grid peak information
            optimal_radius - Optimal filter radius used

        Returns:
            fig - Matplotlib figure containing the plots

        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # ORIGINAL image
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original image with grid')
        axes[0, 0].axis('off')
        
        # DEGRIDDER image
        axes[0, 1].imshow(degridded, cmap='gray')
        title = 'Degridded image'
        axes[0, 1].set_title(title)
        axes[0, 1].axis('off')
        
        # DIFFERENCE image
        diff = np.abs(original - degridded)
        axes[0, 2].imshow(diff, cmap='jet')
        axes[0, 2].set_title('Difference = Original - Degridded')
        axes[0, 2].axis('off')
        
        # FFT of original
        orig_fft_mag = np.log(np.abs(fftshift(fft2(original))) + 1)
        axes[1, 0].imshow(orig_fft_mag, cmap='hot')
        axes[1, 0].set_title('Original FFT (log magnitude)')
        axes[1, 0].axis('off')
        
        # FFT of degridded
        degrid_fft_mag = np.log(np.abs(fftshift(fft2(degridded))) + 1)
        axes[1, 1].imshow(degrid_fft_mag, cmap='hot')
        axes[1, 1].set_title('Degridded FFT (log magnitude)')
        axes[1, 1].axis('off')
        
        # FILTER visualization
        if grid_peaks is not None:
            grid_filter = self.create_grid_filter(grid_peaks, optimal_radius or 5)
            if grid_filter is not None:
                axes[1, 2].imshow(grid_filter, cmap='gray')
                axes[1, 2].set_title('Grid filter applied')
                axes[1, 2].axis('off')
                
                # MARK filtered peaks
                valid_mask = (grid_peaks['frequencies'][:, 2] >= self.params.expected_grid_frequency_range[0]) & \
                            (grid_peaks['frequencies'][:, 2] <= self.params.expected_grid_frequency_range[1])
                valid_peaks = grid_peaks['peak_coords'][valid_mask]
                
                if len(valid_peaks) > 0:
                    axes[1, 2].scatter(valid_peaks[:, 1], valid_peaks[:, 0], 
                                      c='red', s=30, marker='x', alpha=0.8)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    

    def save_degridded_image(self, degridded_image, output_path=None):
        """
        Save the degridded image with automatic filename generation.
        
        Args:
            degridded_image - Degridded image array (normalized 0-1)
            output_path - Output file path (if None, will generate automatic filename)

        Returns:
            output_path - Path where the image was saved (str)

        """
        try:
            # KEEP as float32 with original dynamic range for better ImageJ compatibility
            output_image = degridded_image.copy()
            
            # SCALE back to original range but keep as float
            if hasattr(self, 'img_min') and hasattr(self, 'img_max'):
                output_image = output_image * (self.img_max - self.img_min) + self.img_min
            
            # CONVERT to float32 for optimal ImageJ compatibility and file size
            output_image = output_image.astype(np.float32)
            
            # GENERATE filename if not provided
            if output_path is None:
                base, ext = os.path.splitext(os.path.basename(self.dark_field_path))
                
                # GET current date
                today = datetime.datetime.now()
                date_str = today.strftime("%y%m%d")
                
                # CREATE base filename with date and degrid identifier
                base_filename = f"{base}_{date_str}_DEGRID{ext}"
                
                # GET directory of original file
                output_dir = os.path.dirname(self.dark_field_path)
                base_filepath = os.path.join(output_dir, base_filename)
                
                # FIND unique filename if file already exists
                counter = 1
                output_path = base_filepath
                while os.path.exists(output_path):
                    name, extension = os.path.splitext(base_filepath)
                    output_path = f"{name}_{counter:03d}{extension}"
                    counter += 1
            
            # SAVE image
            if len(output_image.shape) == 2:
                # Use PIL with mode 'F' for 32-bit floating point
                pil_image = Image.fromarray(output_image, mode='F')
            else:
                # For color images (unlikely in this application)
                pil_image = Image.fromarray(output_image)
            
            pil_image.save(output_path)
            print(f"Degridded image saved as float32 TIFF to: {output_path}")
            print(f"Image range: {output_image.min():.6f} to {output_image.max():.6f}")
            # SAVE processing log
            self._save_processing_log(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error saving degridded image: {e}")
            return None
    

    def _save_processing_log(self, image_output_path):
        """
        Save a log file with processing parameters and results.
        
        Args:
            image_output_path - Path where the degridded image was saved

        Returns:
            None - The log file is saved to the same directory as the output image

        """
        try:
            # CREATE log filename
            base_path, ext = os.path.splitext(image_output_path)
            log_path = f"{base_path}_processing_log.txt"
            
            # GET current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # CREATE log content
            log_content = f"""
====================================================                        
PROCESSING LOG
====================================================

Timestamp: {timestamp}
Version: DEGRID 1.0

INPUT FILES:
-----------
Selected image: {os.path.basename(self.dark_field_path)}
Reference image:  {os.path.basename(self.reference_path)}
Clean flat field: {os.path.basename(self.flat_field_path)}

OUTPUT FILE:
-----------
Degridded image: {os.path.basename(image_output_path)}

IMAGE INFORMATION:
-----------------
Original image dtype: {self.orig_dtype if hasattr(self, 'orig_dtype') else 'Unknown'}
Working image dtype: {self.working_image.dtype if hasattr(self, 'working_image') else 'Unknown'}
Image dimensions: {self.working_image.shape if hasattr(self, 'working_image') else 'Unknown'}
Normalization range: {self.img_min} - {self.img_max} 


PARAMETERS
====================================================

OPTIMIZATION:
------------
Method:                 {self.params.optimization_method}
Max iterations:         {self.params.max_iterations}
Function tolerance:     {self.params.ftol}
Gradient tolerance:     {self.params.gtol}
General tolerance:      {self.params.tolerance}

Peak threshold:         {self.params.peak_threshold:.6f}
Min distance:           {self.params.peak_min_distance}
Gaussian sigma:         {self.params.gaussian_sigma:.6f}
Peak threshold bounds:  {self.params.peak_threshold_bounds}
Min distance bounds:    {self.params.peak_min_distance_bounds}
Gaussian sigma bounds:  {self.params.gaussian_sigma_bounds}

GRID DETECTION:
--------------
Frequency range:        {self.params.expected_grid_frequency_range[0]:.3f} - {self.params.expected_grid_frequency_range[1]:.3f}
Max peaks to find:      {self.params.max_peaks_to_find}
DC exclusion radius:    {self.params.exclude_dc_radius}

Base filter radius:        {self.params.base_filter_radius}
Max filter radius:         {self.params.max_filter_radius}
Default filter radius:     {self.params.default_filter_radius}
Quality threshold:         {self.params.quality_threshold}
Confidence peak threshold: {self.params.confidence_peak_threshold}

DEGRID:
------
Nyquist sampling factor:   {self.params.nyquist_sampling_factor}
Gaussian sigma multiplier: {self.params.gaussian_sigma_multiplier}
Statistical confidence:    {self.params.statistical_confidence_level}
Half-max threshold:        {self.params.half_max_threshold}
Angular resolution:        {self.params.angular_resolution_degrees}
Estimator percentile:      {self.params.robust_estimator_percentile}

====================================================
"""
            # WRITE log file
            with open(log_path, 'w') as f:
                f.write(log_content)
            
            print(f"Processing log saved to: {log_path}")
            
        except Exception as e:
            print(f"Warning: Could not save processing log: {e}")
    


    def process_complete_degridding(self, visualize=True, save_result=True, adaptive_filtering=True):
        """
        Removal of grid from selected image.
        
        Args:
            visualize - Whether to show visualizations
            save_result - Whether to prompt for saving the result
            adaptive_filtering - Whether to use adaptive filtering for optimal results

        Returns:
            results:
                grid_peaks - Dictionary containing detected grid peak information
                degridded_image - Image with grid artifacts removed
                optimal_radius - Optimal filter radius used
                quality_score - Quality score of the degridding process
                output_path - Path where the degridded image was saved
            
        """
        results = {
            'grid_peaks': None,
            'degridded_image': None,
            'optimal_radius': None,
            'quality_score': None,
            'output_path': None
        }
        
        try:
            # DETECT grid peaks with optimization
            print("Detecting grid peaks...")
            grid_peaks = self.process_grid_detection(
                visualize=visualize,
                optimize_params=True
            )
            
            if grid_peaks is None:
                print("Grid detection failed!")
                return results
            
            results['grid_peaks'] = grid_peaks
            
            # REMOVE grid from dark field image
            print("\nRemoving grid from dark field image...")
            
            if adaptive_filtering:
                degridded_image, optimal_radius = self.adaptive_grid_removal(
                    self.working_image, 
                    grid_peaks
                )
                results['optimal_radius'] = optimal_radius
            else:
                degridded_image = self.remove_grid_from_image(
                    self.working_image, 
                    grid_peaks
                )
                results['optimal_radius'] = self.params.default_filter_radius
            
            results['degridded_image'] = degridded_image
            
            # EVALUEATE quality
            quality_score = self._evaluate_degridding_quality(
                self.working_image, 
                degridded_image, 
                grid_peaks
            )
            results['quality_score'] = quality_score
            
            # VIZUALISE results
            if visualize:
                print("\nVisualizing results...")
                self.visualize_grid_removal_results(
                    self.working_image, 
                    degridded_image, 
                    grid_peaks, 
                    results['optimal_radius']
                )
            
            # SAVE results
            if save_result:
                print("\nSaving results...")
                # AUTOMATIC filename generation
                output_path = self.save_degridded_image(degridded_image, output_path=None)
                results['output_path'] = output_path
            
            print(f"\nDegridding completed successfully!")
            print(f"Quality score: {quality_score:.4f}")
            if results['optimal_radius']:
                print(f"Optimal filter radius: {results['optimal_radius']}")
            
            return results
            
        except Exception as e:
            print(f"Error in complete degridding process: {str(e)}")
            return results

def run_simple_degridding():
    """
    Run the degridding pipeline.
    """
    try:
        print("="*70)
        print("AUTOMATIC GRID REMOVAL")
        print("="*70)
        print("Starting automatic grid detection and removal process...")
        print()
        
        # INITIALIZE the grid removal system
        grid_remover = GridRemoval()
        results = grid_remover.process_complete_degridding(
            visualize=True,
            save_result=True,
            adaptive_filtering=True
        )
        
        if results['degridded_image'] is not None:
            print("")
            print("\n" + "="*70)
            print("SUCCESS - Grid removal completed!")
            print("="*70)
            # PRINT a wizard with staff
            print(r"""
                  .

                   .
         /^\     .
    /\   "V"
   /__\   I      O  o
  //..\\  I     .
  \].`[/  I
  /l\/j\  (]    .  O
 /. ~~ ,\/I          .
 \\L__j^\/I       o
  \/--v}  I     o   .
  |    |  I   _________
  |    |  I c(`       ')o
  |    l  I   \.     ,/
_/j  L l\_!  _//^---^\\_    -Row
""")
            # PRINT summary of results
            valid_mask = (results['grid_peaks']['frequencies'][:, 2] >= 
                         grid_remover.params.expected_grid_frequency_range[0]) & \
                        (results['grid_peaks']['frequencies'][:, 2] <= 
                         grid_remover.params.expected_grid_frequency_range[1])
            print(f"Grid peaks detected in Fourier space: {np.sum(valid_mask)}")
            print(f"MSE-score: {results['quality_score']:.4f}")
            print(f"Optimized filter radius: {results['optimal_radius']} pixels")
            if results['output_path']:
                print(f"Output saved to: {os.path.basename(results['output_path'])}")
        else:
            print("\nGrid removal failed. Please check your images and try again.")
            
    except Exception as e:
        print(f"Error in degridding: {e}")
        messagebox.showerror("Error", str(e))



# ============== MAIN EXECUTION =======================================================================

def main():

    try:
        print("="*70)
        print("AUTOMATIC GRID REMOVAL SYSTEM")
        print("="*70)
        print("You will need three images:")
        print("1. Image with grid artifacts")
        print("2. Reference image (grid pattern only)")
        print("3. Clean flat field image (without grid)")
        print()
        
        run_simple_degridding()
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    main()

