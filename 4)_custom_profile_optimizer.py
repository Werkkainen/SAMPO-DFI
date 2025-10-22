import numpy as np
from PIL import Image
from bm3d import bm3d, BM3DProfile
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from scipy.optimize import minimize_scalar
import time


#####################################################################
#                                                                   #
#  Custom profile optimizer for Block Matching 3D (BM3D) Denoising  #
#                                                                   #
#####################################################################

# ==================================================================================
# DESCRIPTION:
# > Select via file dialog: 
#   - Input image with streaks to denoise.
#   - Interactive ROI selection: click and drag to select region...
# > Creates a Gaussian-blurred ground truth reference.
# > Optimizes BM3D profile parameters for streak removal using the ROI.
# > Saves the optimized profile to a Python file with date stamp for reuse.
# ==================================================================================

"""
This script creates an optimized BM3D profile for streak removal in images,
which can be used in the main BM3D script for streak removal. This is recommended
to be used as a one-time calibration for the setup used as it is computationally expensive.

The BM3D library is based on:
Y. Mäkinen, L. Azzari, A. Foi, 2020, "Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching", in IEEE Transactions on Image Processing, vol. 29, pp. 8339-8354.
K. Dabov, A. Foi, V. Katkovnik, K. Egiazarian, 2007, "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering", in IEEE Transactions on Image Processing, vol. 16, pp. 2080-2095.

Please check the requirements.txt file for the complete list of dependencies and their versions.
Python           3.12.0
bm3d             4.0.3    
bm4d             4.2.5 
opencv           4.11.0.86
matplotlib       3.10.3
numpy            2.3.1
pillow           11.3.0
scipy            1.16.0
scikit-image     0.25.2

Date: 21.10.2025
Author: Werneri A. Lindberg
Acknowledgment: Some parts of the script were written with assistance from GitHub Copilot.

"""


# ============= CUSTOM BM3D PROFILE PARAMETERS ==========================================================
'''
These parameters are iterated to find the best profile parameters for streak removal of your images.
'''

def create_custom_streak_profile(bs_ht=8, bs_wiener=8, max_3d_size_ht=16, max_3d_size_wiener=32, 
                                lambda_thr3d=2.7, tau_match=2500, tau_match_wiener=400,
                                transform_2d_ht='dct', transform_2d_wiener='dct',
                                transform_3rd_dim='haar'):
    """
    Create a custom BM3D profile optimized for streak removal in images.
    
    
    BLOCK SIZES - How big the image patches are.
    --------------------------------------------
    bs_ht: Block size for hard thresholding (first pass)
      - Smaller blocks = Better for fine, thin streaks
      - Larger blocks = Better for thick, coarse streaks
    bs_wiener: Block size for Wiener filtering (second pass)
    
    GROUPING LIMITS - How many similar patches to group together.
    -------------------------------------------------------------
    max_3d_size_ht: Max patches in hard thresholding group
      - More patches = Better streak removal but slower processing
      - Fewer patches = Faster but may miss some streaks
    max_3d_size_wiener: Max patches in Wiener filtering group
    
    NOISE THRESHOLD - How aggressive the denoising is.
    --------------------------------------------------
    lambda_thr3d: 3D transform shrinkage threshold
      - Lower values = Gentle denoising, preserves details
      - Higher values = Aggressive streak removal
    
    SIMILARITY MATCHING - How strict the patch matching is.
    -------------------------------------------------------
    tau_match: Distance threshold for hard thresholding
      - Lower values = Stricter matching, fewer but better patches
      - Higher values = More forgiving, includes more patches
    tau_match_wiener: Distance threshold for Wiener filtering
    
    TRANSFORM TYPES - Mathematical operations used.
    -----------------------------------------------
    transform_2d_ht ('dct'): 2D transform for hard thresholding
      - 'dct' = Discrete Cosine Transform. Good for most streak patterns (recommended).
    transform_2d_wiener ('dct'): 2D transform for Wiener filtering
    transform_3rd_dim ('haar'/'dct'): Transform across grouped patches
      - 'haar' = Haar Transform.
      - 'dct' = Discrete Cosine Transform.
    
    """
    profile = BM3DProfile()
    
    # BLOCK SIZES
    profile.bs_ht = bs_ht  
    profile.bs_wiener = bs_wiener
    
    # GROUP SIZES - control how many similar patches are processed together
    profile.max_3d_size_ht = max_3d_size_ht
    profile.max_3d_size_wiener = max_3d_size_wiener
    
    # THRESHOLD - how aggressive the denoising is
    if lambda_thr3d is not None:
        profile.lambda_thr3d = lambda_thr3d

    # SIMILARITY MATCHING - determines which patches are considered "similar"
    profile.tau_match = tau_match
    profile.tau_match_wiener = tau_match_wiener
    
    # TRANSFORM TYPES - mathematical operations for denoising
    profile.transform_2d_ht_name = transform_2d_ht
    profile.transform_2d_wiener_name = transform_2d_wiener
    profile.transform_3rd_dim_name = transform_3rd_dim
    
    return profile

# ============= PROCESSING PARAMETERS =======================================================
class ProcessingParams:

    def __init__(self):
        self.passes = 2                   # Number of BM3D passes for higher quality denoising
        self.gaussian_sigma = 15.0        # Overexaggerated averaging for ground truth
        self.psd_range = (0.1, 2.0)       # Range for Power Spectral Density (PSD) optimization
        self.optimization_metric = 'mse'  # Mean Squared Error (MSE) for optimization
        self.use_roi = True               # Use ROI for optimization to make it faster
        self.profile_optimization_rounds = 3  # Number of optimization rounds

# ============= THE DENOISE CLASS =======================================================
class Denoise:
    def __init__(self, params: ProcessingParams):
        self.params = params
        self.img = None
        self.img_np = None
        self.img_norm = None 
        self.img_min = None
        self.img_max = None
        self.denoised = None
        self.ground_truth = None
        self.optimized_denoised = None
        self.optimal_psd = None
        self.optimization_results = []
        self.roi_coords = None
        self.roi_img = None
        self.roi_ground_truth = None
        self.current_profile = None
        self.best_profile_params = None
        self.profile_optimization_results = []


    def select_image(self, title):
        Tk().withdraw()
        selected_file = filedialog.askopenfilename(title=title, filetypes=[("TIFF files", "*.tif;*.tiff"), ("All Files", "*.*")])
        if not selected_file:
            print(f"User canceled the file selection for {title}.")
            exit()
        if not os.path.isfile(selected_file):
            raise FileNotFoundError(f"File not found: {selected_file}")
        return selected_file


    def load_image(self, file_path):
        img = Image.open(file_path)
        img_np = np.array(img, dtype=np.float32)
        if len(img_np.shape) > 2:
            raise ValueError("The selected image is not grayscale.")
        return img, img_np

    def create_ground_truth(self, img_norm):
        """Create a Gaussian-blurred version as ground truth reference"""
        ground_truth = gaussian_filter(img_norm, sigma=self.params.gaussian_sigma)
        return ground_truth

    def select_roi_interactive(self, img_norm):
        """Interactive ROI selection using matplotlib"""
        print("Click and drag to select ROI for optimization...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_norm, cmap='gray')
        ax.set_title('Click and drag to select ROI for optimization')
        
        roi_selected = {'coords': None}
        
        def on_click(event):
            if event.inaxes != ax:
                return
            roi_selected['x1'] = int(event.xdata)
            roi_selected['y1'] = int(event.ydata)
        
        def on_release(event):
            if event.inaxes != ax:
                return
            roi_selected['x2'] = int(event.xdata)
            roi_selected['y2'] = int(event.ydata)
            
            # ENSURE proper ordering
            x1, x2 = sorted([roi_selected['x1'], roi_selected['x2']])
            y1, y2 = sorted([roi_selected['y1'], roi_selected['y2']])
            
            self.roi_coords = (y1, y2, x1, x2)
            roi_selected['coords'] = self.roi_coords
            
            # DRAW rectangle
            ax.clear()
            ax.imshow(img_norm, cmap='gray')
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(f'ROI selected: ({x1}, {y1}) to ({x2}, {y2}), size: {x2-x1}x{y2-y1}')
            plt.draw()
            
            print(f"ROI selected: ({x1}, {y1}) to ({x2}, {y2}), size: {x2-x1}x{y2-y1}")
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('button_release_event', on_release)
        
        plt.show()
        
        if roi_selected['coords'] is None:
            print("No ROI selected, please select a valid ROI region")
            return None
        
        y1, y2, x1, x2 = self.roi_coords
        return img_norm[y1:y2, x1:x2]

    def calculate_metric(self, denoised, ground_truth, metric='mse'):
        """Calculate similarity metric between denoised image and ground truth"""
        if denoised is None or ground_truth is None:
            print(f"Warning: Cannot calculate metric - denoised or ground_truth is None")
            return float('inf') if metric == 'mse' else -float('inf')
        
        # ENSURE both arrays are valid numpy arrays
        if not isinstance(denoised, np.ndarray) or not isinstance(ground_truth, np.ndarray):
            print(f"Warning: Invalid array types - denoised: {type(denoised)}, ground_truth: {type(ground_truth)}")
            return float('inf') if metric == 'mse' else -float('inf')
        
        # CHECK for NaN or inf values
        if np.any(np.isnan(denoised)) or np.any(np.isinf(denoised)):
            print(f"Warning: Denoised image contains NaN or inf values")
            return float('inf') if metric == 'mse' else -float('inf')
        
        try:
            if metric == 'ssim':
                # SSIM ranges from -1 to 1, where 1 is perfect similarity (higher is better)
                return ssim(ground_truth, denoised, data_range=1.0)
            elif metric == 'psnr':
                # PSNR in dB, higher is better
                return psnr(ground_truth, denoised, data_range=1.0)
            elif metric == 'mse':
                # MSE, lower is better (return positive value)
                return np.mean((ground_truth - denoised) ** 2)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        except Exception as e:
            print(f"Error calculating {metric}: {e}")
            return float('inf') if metric == 'mse' else -float('inf')

    def get_profile(self):
        """Get the appropriate BM3D profile based on settings"""
        if self.current_profile is not None:
            return self.current_profile
        else:
            # BACKUP: Return default if no optimization done yet
            return 'np'


    def denoise_with_psd(self, img_norm, psd_value, roi_only=False, custom_profile=None):
        """Apply BM3D denoising with specific PSD value and profile"""
        
        # GET the appropriate profile
        if custom_profile is not None:
            profile = custom_profile
        else:
            profile = self.get_profile()
        
        try:
            if roi_only and self.roi_coords is not None:
                # ONLY denoise the ROI for optimization
                y1, y2, x1, x2 = self.roi_coords
                roi_img = img_norm[y1:y2, x1:x2].copy()
                current_norm = roi_img.copy()
                for i in range(self.params.passes):
                    result = bm3d(current_norm, psd_value, profile=profile)
                    if result is None:
                        print(f"Warning: BM3D pass {i+1} returned None")
                        return None
                    current_norm = result
                return current_norm
            else:
                # DENOISE the full image
                current_norm = img_norm.copy()
                for i in range(self.params.passes):
                    result = bm3d(current_norm, psd_value, profile=profile)
                    if result is None:
                        print(f"Warning: BM3D pass {i+1} returned None")
                        return None
                    current_norm = result
                return current_norm
        except Exception as e:
            print(f"Error in denoise_with_psd: {e}")
            return None


    def evaluate_profile_performance(self, profile_params, psd_value=None):
        """Evaluate the performance of a specific profile configuration"""
        try:
            # USE provided PSD or optimal one
            if psd_value is None:
                psd_value = self.optimal_psd if hasattr(self, 'optimal_psd') and self.optimal_psd is not None else 1.0
            
            # CHECK if we have valid image data
            if self.img_norm is None:
                print(f"Error: self.img_norm is None")
                return float('inf') if self.params.optimization_metric == 'mse' else -float('inf')
            
            #  CREATE profile from parameters
            custom_profile = create_custom_streak_profile(**profile_params)
            
            if self.params.use_roi and self.roi_coords is not None:
                # USE ROI for fast evaluation
                if self.roi_ground_truth is None:
                    print(f"Error: self.roi_ground_truth is None")
                    return float('inf') if self.params.optimization_metric == 'mse' else -float('inf')
                    
                roi_denoised = self.denoise_with_psd(self.img_norm, psd_value, roi_only=True, custom_profile=custom_profile)
                if roi_denoised is None:
                    print(f"Warning: BM3D returned None for ROI denoising")
                    return float('inf') if self.params.optimization_metric == 'mse' else -float('inf')
                
                print(f"roi_denoised shape: {roi_denoised.shape}, roi_ground_truth shape: {self.roi_ground_truth.shape}")
                metric_value = self.calculate_metric(roi_denoised, self.roi_ground_truth, self.params.optimization_metric)
            else:
                # USE full image
                if self.ground_truth is None:
                    print(f"Error: self.ground_truth is None")
                    return float('inf') if self.params.optimization_metric == 'mse' else -float('inf')
                    
                denoised = self.denoise_with_psd(self.img_norm, psd_value, roi_only=False, custom_profile=custom_profile)
                if denoised is None:
                    print(f"Warning: BM3D returned None for full image denoising")
                    return float('inf') if self.params.optimization_metric == 'mse' else -float('inf')
                    
                print(f"denoised shape: {denoised.shape}, ground_truth shape: {self.ground_truth.shape}")
                metric_value = self.calculate_metric(denoised, self.ground_truth, self.params.optimization_metric)
            
            return metric_value
            
        except Exception as e:
            print(f"Error evaluating profile: {e}")
            import traceback
            traceback.print_exc()
            return float('inf') if self.params.optimization_metric == 'mse' else -float('inf')


    def optimize_profile_parameters(self):
        """Optimize BM3D profile parameters for streak removal"""
        print("Starting profile parameter optimization for streak removal...")
        print("=" * 60)
        
        best_metric = float('inf') if self.params.optimization_metric == 'mse' else -float('inf')
        best_params = None
        
        # RANGES for optimization
        param_ranges = {
            'bs_ht': [4, 6, 8, 12, 16],
            'bs_wiener': [4, 6, 8, 12, 16],
            'max_3d_size_ht': [16, 24, 32, 40, 48],
            'max_3d_size_wiener': [32, 48, 64, 80, 96],
            'lambda_thr3d': [2.0, 2.5, 2.7, 3.0, 3.2, 3.5, 4.0],
            'tau_match': [1200, 1500, 2000, 2500, 3000, 3500],
            'tau_match_wiener': [250, 300, 350, 400, 500, 600],
            # You may try other transforms as well if the version of BM3D supports them.
            # The Discrete Cosine Transform (DCT) is recommended for streak removal.
            # Note that the optimization takes longer with more transforms to be tested.
            'transform_2d_ht': ['dct'],
            'transform_2d_wiener': ['dct'],
            'transform_3rd_dim': ['haar', 'dct']
        }
        
        # INITIAL guessing of parameters
        base_configs = [
            # Fine streaks config - small blocks, aggressive thresholding
            {'bs_ht': 4, 'bs_wiener': 4, 'max_3d_size_ht': 32, 'max_3d_size_wiener': 64, 'lambda_thr3d': 3.2, 
             'tau_match': 1500, 'tau_match_wiener': 300, 'transform_2d_ht': 'dct', 
             'transform_2d_wiener': 'dct', 'transform_3rd_dim': 'dct'},
            # Medium streaks config - balanced approach
            {'bs_ht': 8, 'bs_wiener': 8, 'max_3d_size_ht': 24, 'max_3d_size_wiener': 48, 'lambda_thr3d': 2.8,
             'tau_match': 2000, 'tau_match_wiener': 350, 'transform_2d_ht': 'dct',
             'transform_2d_wiener': 'dct', 'transform_3rd_dim': 'haar'},
            # Coarse streaks config - large blocks, moderate thresholding
            {'bs_ht': 16, 'bs_wiener': 16, 'max_3d_size_ht': 16, 'max_3d_size_wiener': 32, 'lambda_thr3d': 2.5,
             'tau_match': 3000, 'tau_match_wiener': 500, 'transform_2d_ht': 'dct',
             'transform_2d_wiener': 'dct', 'transform_3rd_dim': 'haar'},
            # Aggressive config - maximum streak suppression
            {'bs_ht': 6, 'bs_wiener': 6, 'max_3d_size_ht': 40, 'max_3d_size_wiener': 80, 'lambda_thr3d': 3.5,
             'tau_match': 1200, 'tau_match_wiener': 250, 'transform_2d_ht': 'dct',
             'transform_2d_wiener': 'dct', 'transform_3rd_dim': 'dct'}
        ]
        
        print(f"Testing {len(base_configs)} base configurations...")
        
        # TEST base configurations
        for i, config in enumerate(base_configs):
            print(f"\nTesting base config {i+1}/{len(base_configs)}: {config['bs_ht']}x{config['bs_ht']} blocks, lambda={config['lambda_thr3d']}")
            metric_value = self.evaluate_profile_performance(config)
            
            self.profile_optimization_results.append({
                'config': config.copy(),
                'metric_value': metric_value,
                'type': f'base_config_{i+1}'
            })
            
            is_better = (metric_value < best_metric) if self.params.optimization_metric == 'mse' else (metric_value > best_metric)
            if is_better:
                best_metric = metric_value
                best_params = config.copy()
                print(f"  -> NEW BEST: {self.params.optimization_metric.upper()}={metric_value:.6f}")
            else:
                print(f"  -> {self.params.optimization_metric.upper()}={metric_value:.6f}")
        
        # REFINEMENT ROUNDS: optimize around the best configuration
        if best_params is None:
            print("No valid base configuration found. Cannot proceed with refinement.")
            return None
            
        print(f"\nStarting {self.params.profile_optimization_rounds} refinement rounds...")
        current_best = best_params.copy()
        
        for round_num in range(self.params.profile_optimization_rounds):
            print(f"\n--- Refinement Round {round_num + 1}/{self.params.profile_optimization_rounds} ---")
            round_improved = False
            
            # TRY variations of each parameter
            for param_name, param_values in param_ranges.items():
                if param_name not in current_best:
                    continue
                    
                current_value = current_best[param_name]
                
                # TEST neighboring values
                for new_value in param_values:
                    if new_value == current_value:
                        continue
                    
                    test_config = current_best.copy()
                    test_config[param_name] = new_value
                    
                    print(f"  Testing {param_name}={new_value} (was {current_value})")
                    metric_value = self.evaluate_profile_performance(test_config)
                    
                    self.profile_optimization_results.append({
                        'config': test_config.copy(),
                        'metric_value': metric_value,
                        'type': f'round_{round_num+1}_{param_name}'
                    })
                    
                    is_better = (metric_value < best_metric) if self.params.optimization_metric == 'mse' else (metric_value > best_metric)
                    if is_better:
                        best_metric = metric_value
                        best_params = test_config.copy()
                        current_best = test_config.copy()
                        round_improved = True
                        print(f"    -> IMPROVED: {self.params.optimization_metric.upper()}={metric_value:.6f}")
                    else:
                        print(f"    -> {self.params.optimization_metric.upper()}={metric_value:.6f}")
            
            if not round_improved:
                print(f"  No improvement in round {round_num + 1}, stopping refinement.")
                break
        
        if best_params is None:
            print("ERROR: No valid profile configuration found!")
            return None
            
        self.best_profile_params = best_params
        self.current_profile = create_custom_streak_profile(**best_params)
        
        print(f"\n" + "=" * 60)
        print("PROFILE OPTIMIZATION COMPLETE")
        print(f"Best {self.params.optimization_metric.upper()}: {best_metric:.6f}")
        print("Optimal profile parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print("=" * 60)
        
        return best_params


    def save_optimized_profile(self, filename=None):
        """Save the optimized profile parameters to a Python file for reuse, with date stamp in filename"""
        import datetime
        if self.best_profile_params is None:
            print("No optimized profile to save. Run optimization first.")
            return
        # ADD date stamp to filename (in front)
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        base_name = "optimized_streak_profile.py" if filename is None else filename
        if base_name.endswith('.py'):
            base_name = base_name[:-3]
        filename = f"{date_str}_{base_name}.py"

        # IF AVAILABLE, include the optimized PSD in the generated file
        psd_value = getattr(self, 'optimal_psd', None)

        profile_code = f'''# Optimized BM3D Profile for Streak Removal
# Generated by Custom Profile Optimizer
# Optimization metric: {self.params.optimization_metric.upper()}
# Best score: {self.profile_optimization_results[-1]["metric_value"]:.6f}
'''

        if psd_value is not None:
            profile_code += f"\n# Optimized PSD found during calibration\nOPTIMAL_PSD = {psd_value:.6f}\n\n"
            profile_code += (
                "def get_optimal_psd():\n"
                "    return OPTIMAL_PSD\n\n"
                "def get_optimized_psd():\n"
                "    return OPTIMAL_PSD\n\n"
            )

        profile_code += '''from bm3d import BM3DProfile

def create_optimized_streak_profile():
    """Create the optimized BM3D profile for streak removal"""
    profile = BM3DProfile()
    
    # Optimized parameters
'''
        for param, value in self.best_profile_params.items():
            if isinstance(value, str):
                profile_code += f"    profile.{param}_name = '{value}'\n"
            else:
                profile_code += f"    profile.{param} = {value}\n"
        profile_code += '''
    # Note: Standard parameters (nf, k, p, etc.) are omitted to ensure
    # compatibility with the installed bm3d version.
    
    return profile

# Usage example:
# from bm3d import bm3d
# profile = create_optimized_streak_profile()
# denoised = bm3d(noisy_image, sigma_psd, profile=profile)
'''
        with open(filename, 'w') as f:
            f.write(profile_code)
        print(f"Optimized profile saved to: {filename}")
        return filename  # Return the filename for use in main()


    def run_calibration(self):
        """Run calibration: optimize BM3D profile parameters and PSD, then save a profile file."""
        # LOAD the image
        print("Loading image for BM3D calibration...")
        file_path = self.select_image("Select the image to denoise")
        self.img, self.img_np = self.load_image(file_path)
        self.img_min, self.img_max = self.img_np.min(), self.img_np.max()
        self.img_norm = (self.img_np - self.img_min) / (self.img_max - self.img_min)
        
        # CREATE ground truth reference
        print("Creating ground truth reference using exaggerated averaging...")
        self.ground_truth = self.create_ground_truth(self.img_norm)
        # DISPLAY ground truth range and original image range
        gt_min, gt_max = self.ground_truth.min(), self.ground_truth.max()
        print(f"Ground truth range: [{gt_min:.8f}, {gt_max:.8f}]")
        print(f"Original image range: [{self.img_norm.min():.8f}, {self.img_norm.max():.8f}]")
        
        # SELECT ROI if enabled
        if self.params.use_roi:
            print("ROI optimization enabled.")
            print("Interactive ROI selection - click and drag to select region...")
            self.roi_img = self.select_roi_interactive(self.img_norm)
            if self.roi_img is None:
                print("No valid ROI selected. Exiting...")
                return
            # STORE ROI coordinates
            y1, y2, x1, x2 = self.roi_coords
            self.roi_ground_truth = self.ground_truth[y1:y2, x1:x2]
        else:
            print("Using full image for optimization (slower).")
        
        # OPTIMIZE profile parameters
        print(f"\n{'='*60}")
        print("OPTIMIZING BM3D PROFILE FOR STREAK REMOVAL")
        print(f"{'='*60}")
        profile_start_time = time.time()
        optimization_result = self.optimize_profile_parameters()
        profile_time = time.time() - profile_start_time
        
        if optimization_result is None:
            print("Profile optimization failed. Cannot continue.")
            return
            
        print(f"Profile optimization completed in {profile_time:.2f} seconds")
        
        # Step 2: Optimize PSD parameter
        print(f"\n{'='*60}")
        print("STEP 2: OPTIMIZING PSD PARAMETER")
        print(f"{'='*60}")
        
        start_time = time.time()
        self.optimal_psd = self.optimize_psd()
        
        optimization_time = time.time() - start_time
        print(f"PSD optimization completed in {optimization_time:.2f} seconds")
        
        # Step 3: Apply optimized parameters to full image
        #print(f"\n{'='*60}")
        #print("STEP 3: APPLYING OPTIMIZED PARAMETERS TO FULL IMAGE")
        #print(f"{'='*60}")
        
        #print(f"Applying optimized PSD ({self.optimal_psd:.4f}) to full image...")
        #print("Using optimized custom profile")
        
        #self.optimized_denoised = self.denoise_with_psd(self.img_norm, self.optimal_psd, roi_only=False)
        
        # Convert back to original scale
        #self.optimized_denoised_original_scale = self.optimized_denoised * (self.img_max - self.img_min) + self.img_min
        #self.ground_truth_original_scale = self.ground_truth * (self.img_max - self.img_min) + self.img_min
        
        # Clip values
        #self.optimized_denoised_original_scale = np.clip(self.optimized_denoised_original_scale, self.img_min, self.img_max)
        
        # SAVE optimized profile
        saved_filename = self.save_optimized_profile("optimized_streak_profile.py")
        
        # STORE the filename for later use
        self.saved_profile_filename = saved_filename


    def objective_function(self, psd_value):
        """Objective function for PSD optimization using ROI"""
        try:
            if self.params.use_roi and self.roi_coords is not None:
                # USE ROI for fast optimization
                roi_denoised = self.denoise_with_psd(self.img_norm, psd_value, roi_only=True)
                metric_value = self.calculate_metric(roi_denoised, self.roi_ground_truth, self.params.optimization_metric)
            else:
                # USE full image (slower)
                denoised = self.denoise_with_psd(self.img_norm, psd_value, roi_only=False)
                metric_value = self.calculate_metric(denoised, self.ground_truth, self.params.optimization_metric)
            
            # STORE results for analysis
            self.optimization_results.append({
                'psd': psd_value,
                'metric_value': metric_value,
                'metric_name': self.params.optimization_metric
            })
            
            print(f"PSD: {psd_value:.3f}, {self.params.optimization_metric.upper()}: {metric_value:.6f}")
            
            # For minimize_scalar: 
            # - SSIM and PSNR: higher is better, so return negative to minimize
            # - MSE: lower is better, so return positive to minimize
            if self.params.optimization_metric in ['ssim', 'psnr']:
                objective_value = -metric_value  # RETURN NEGATIVE because we want to maximize these
            else:  # MSE
                objective_value = metric_value   # RETURN POSITIVE because we want to minimize this

            print(f"    -> Objective value returned to optimizer: {objective_value:.6f}")
            return objective_value
            
        except Exception as e:
            print(f"Error with PSD {psd_value}: {e}")
            return float('inf')


    def optimize_psd(self):
        """Optimize PSD parameter using the ground truth reference"""
        roi_text = "ROI-based" if self.params.use_roi else "full image"
        print(f"Optimizing PSD using {self.params.optimization_metric.upper()} metric ({roi_text})...")
        print(f"PSD search range: {self.params.psd_range}")
        
        # CLEAR previous results
        self.optimization_results = []
        
        # OPTIMIZE using scipy
        result = minimize_scalar(
            self.objective_function,
            bounds=self.params.psd_range,
            method='bounded',
            options={'disp': True}
        )
        
        self.optimal_psd = result.x
        print(f"\nOptimal PSD found: {self.optimal_psd:.4f}")
        
        return self.optimal_psd


# ============== MAIN EXECUTION ==============================================================

def main():
    params = ProcessingParams()
    denoiser = Denoise(params)
    
    print("BM3D Custom Profile Optimizer for Streak Removal")
    print("=" * 60)
    print("CALIBRATION MODE: Optimize BM3D profiles for your specific streak patterns")
    print("=" * 60)
    print(f"Gaussian sigma for ground truth: {params.gaussian_sigma}")
    print(f"PSD search range: {params.psd_range}")
    print(f"Optimization metric: {params.optimization_metric}")
    print(f"BM3D passes: {params.passes}")
    print(f"Use ROI for optimization: {params.use_roi}")
    print(f"Optimization rounds: {params.profile_optimization_rounds}")
    print()
    
    denoiser.run_calibration()
    
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE!")
    print("=" * 60)
    print("[+] Custom streak removal profile optimized")
    if hasattr(denoiser, 'saved_profile_filename') and denoiser.saved_profile_filename:
        print(f"[+] Profile saved to '{denoiser.saved_profile_filename}'")
    else:
        print("[+] Profile saved")
    print("[+] You can now use this profile in your main BM3D filter script")
    print()
    if hasattr(denoiser, 'optimal_psd') and denoiser.optimal_psd is not None:
        print(f"4. Optimal PSD found: {denoiser.optimal_psd:.4f}")
    else:
        print("4. Optimal PSD: Not found (optimization failed)")

if __name__ == "__main__":
    main()