import numpy as np
from skimage import img_as_float
from scipy.ndimage import uniform_filter, sobel, gaussian_filter
import imageio.v2 as imageio
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import os
from scipy.optimize import minimize_scalar
import tifffile

#########################################
#                                       #
# STAIRCASE REMOVAL USING GUIDED FILTER #
#                                       #
#########################################
"""
Simple automated guided filter for removing staircase artifacts 
from X-ray dark-field images using the reference Transmission image as the guide.

INPUT:
- Choose via file dialog: Dark-field image with staircase artifacts (TIFF)
- Choose via file dialog: Transmission image for guidance (TIFF)

OUTPUT:
- Processed image with staircase artifacts removed
- Multichannel heatmap visualization saved as TIFF for ImageJ compatibility
- Log-file with processing details

Date: 14.8.2025
Author: Werneri A. Lindberg
Acknowledgment: The code was written with assistance from GitHub Copilot.

"""


# ============= PROCESSING PARAMETERS ========================================================
class ProcessingParams:
    
    def __init__(self):
        # GUIDED FILTER PARAMETERS
        # ========================
        self.DEFAULT_RADIUS = 6              # Default filter radius
        self.DEFAULT_REGULARIZATION = 1e-3   # Default regularization parameter
        
        # OPTIMIZATION PARAMETERS
        # ======================
        self.OPTIMIZE_BY_DEFAULT = True           # Enable/disable parameter optimization
        self.RADIUS_RANGE = (3, 15)               # Range for radius optimization
        self.REGULARIZATION_RANGE = (1e-4, 1e-1)  # Range for regularization optimization
        
        # HEATMAP VISUALIZATION PARAMETERS
        # ================================
        self.ENABLE_EDGE_HEATMAP = True              # Enable/disable multichannel heatmap generation
        self.HEATMAP_EDGE_THRESHOLD_PERCENTILE = 85  # Percentile for edge detection threshold
        self.HEATMAP_ALPHA_BLEND = 0.3             # Edge overlay blend intensity (0.0-1.0)
        self.HEATMAP_GAUSSIAN_SIGMA = 1.0          # Smoothing for edge transitions
        
        # FILE NAMING PARAMETERS
        # ======================
        self.DATE_FORMAT = "%y%m%d"                # Date format for filenames
        self.RESULT_SUFFIX = "stair"               # Suffix for result files
        self.HEATMAP_PREFIX = "heatmap_"           # Prefix for heatmap files

# ============================================================================================


def guided_filter(I, p, r=6, eps=1e-3):
    """
    Reference image (transmission) guided filter to suppress staircases.
    
    I   : Guide image (clean transmission) - shows where true edges should be
    p   : Input image (dark-field with artifacts) - the image we want to clean  
    r   : Filter radius - larger values = more smoothing
    eps : Regularization to prevent division by zero
    
    HOW IT WORKS:
    =============
    The algorithm assumes locally the output should follow: output = a*guide + b
    where 'a' and 'b' are coefficients calculated for each local region.
    
    In edge regions (high variance): 'a' is strong → preserves details
    In smooth regions (low variance): 'a' is weak → applies smoothing
    
    """
    win = 2*r+1                                   # Processing window size
    
    # CALCULATE local averages in the processing window
    mean_I = uniform_filter(I, size=win)          # Average intensity of guide image locally
    mean_p = uniform_filter(p, size=win)          # Average intensity of input image locally
    mean_Ip = uniform_filter(I * p, size=win)     # Average of guide×input product locally
    
    # CALCULATE how images vary together (covariance) and separately (variance)
    cov_Ip = mean_Ip - mean_I * mean_p            # Covariance: how guide and input change together
    mean_II = uniform_filter(I * I, size=win)     # Average of guide image squared
    var_I = mean_II - mean_I * mean_I             # Variance: how much guide varies locally
                                                  # High variance = edge region, Low variance = smooth region
    
    # CALCULATE linear coefficients for the relationship output = a*guide + b
    a = cov_Ip / (var_I + eps)                    # Slope coefficient - how strongly output follows guide
                                                  # eps prevents division by zero in smooth regions
    b = mean_p - a * mean_I                       # Offset coefficient - baseline intensity level
    
    # SMOOTH the coefficients to prevent abrupt changes
    mean_a = uniform_filter(a, size=win)          # Smoothed slope values
    mean_b = uniform_filter(b, size=win)          # Smoothed offset values
    
    # APPLY the linear relationship to get final result
    q = mean_a * I + mean_b                       # Final output: smoothed_slope × guide + smoothed_offset
    
    # WHY THIS WORKS FOR STAIRCASE REMOVAL:
    # =====================================
    # In artifact regions: guide is smooth → low variance → more smoothing → artifacts removed
    # In true edge regions: guide has edges → high variance → less smoothing → edges preserved
    # The filter adapts automatically based on the guide image content.

    return q


def compute_edge_similarity(img1, img2):
    """
    Compute edge similarity between two images using Sobel gradients.
    Higher values indicate better edge matching.
    """
    # SOBEL gradients
    grad_x1, grad_y1 = sobel(img1, axis=0), sobel(img1, axis=1)
    grad_x2, grad_y2 = sobel(img2, axis=0), sobel(img2, axis=1)

    # MAGNITUDES
    mag1 = np.sqrt(grad_x1**2 + grad_y1**2)
    mag2 = np.sqrt(grad_x2**2 + grad_y2**2)
    
    # NORMALIZE and prevent division by zero
    mag1_norm = mag1 / (np.max(mag1) + 1e-8)
    mag2_norm = mag2 / (np.max(mag2) + 1e-8)
    
    # CORRELATION coefficient between edge magnitudes
    correlation = np.corrcoef(mag1_norm.flatten(), mag2_norm.flatten())[0, 1]
    
    return correlation if not np.isnan(correlation) else 0.0


def optimize_guided_filter_params(guide_img, input_img, r_range=(3, 15), eps_range=(1e-4, 1e-1)):
    """
    Optimize guided filter parameters by maximizing edge similarity with guide image.
    
    """
    print("Optimizing guided filter parameters...")
    
    # INITIAL guesses for radius and eps
    best_r = 6
    best_eps = 1e-3

    # OPTIMIZE radius (r)
    def objective_r(r):
        r_int = max(1, int(round(r)))
        filtered = guided_filter(guide_img, input_img, r=r_int, eps=1e-3)
        score = compute_edge_similarity(filtered, guide_img)
        return -score
    result_r = minimize_scalar(objective_r, bounds=r_range, method='bounded')
    best_r = max(1, int(round(result_r.x)))
    
    # OPTIMIZE eps with the best radius
    def objective_eps(log_eps):
        eps = 10**log_eps
        filtered = guided_filter(guide_img, input_img, r=best_r, eps=eps)
        score = compute_edge_similarity(filtered, guide_img)
        return -score
    log_eps_range = (np.log10(eps_range[0]), np.log10(eps_range[1]))
    result_eps = minimize_scalar(objective_eps, bounds=log_eps_range, method='bounded')
    best_eps = 10**result_eps.x
    
    # FINAL EVALUATION
    final_filtered = guided_filter(guide_img, input_img, r=best_r, eps=best_eps)
    final_score = compute_edge_similarity(final_filtered, guide_img)
    
    print(f"Optimization complete: r={best_r}, eps={best_eps:.2e}, edge_similarity={final_score:.4f}")
    
    return best_r, best_eps


class ImageProcessor:
    """Image processor that applies guided filter to remove staircase artifacts."""
    
    def __init__(self, image_path=None, transmission_path=None):
        self.params = ProcessingParams()
        
        if image_path is None or transmission_path is None:
            root = tk.Tk()
            root.withdraw()
            
            # SELECT dark-field image
            art_file = filedialog.askopenfilename(
                title="Select dark-field image with staircase artifacts", 
                filetypes=[("TIFF files", "*.tif *.tiff"), ("All Files", "*.*")]
            )
            if not art_file:
                raise Exception("User canceled the dark-field image selection.")
            
            # SELECT transmission image
            trans_file = filedialog.askopenfilename(
                title="Select transmission image for guidance", 
                filetypes=[("TIFF files", "*.tif *.tiff"), ("All Files", "*.*")]
            )
            if not trans_file:
                raise Exception("User canceled the transmission image selection.")
            
            self.image_path = art_file
            self.transmission_path = trans_file
        else:
            self.image_path = image_path
            self.transmission_path = transmission_path
        
        self.load_images()
    
    def load_images(self):
        """Load and normalize images to float range 0-1"""
        # LOAD dark-field image (with artifacts)
        dark_field_raw = imageio.imread(self.image_path)
        if len(dark_field_raw.shape) > 2:
            dark_field_raw = dark_field_raw[:,:,0]
        self.dark_field = img_as_float(dark_field_raw)
        
        # LOAD transmission image (guide)
        transmission_raw = imageio.imread(self.transmission_path)
        if len(transmission_raw.shape) > 2:
            transmission_raw = transmission_raw[:,:,0]
        self.transmission = img_as_float(transmission_raw)
        
        # ENSURE both images have the same dimensions
        if self.dark_field.shape != self.transmission.shape:
            min_h = min(self.dark_field.shape[0], self.transmission.shape[0])
            min_w = min(self.dark_field.shape[1], self.transmission.shape[1])
            self.dark_field = self.dark_field[:min_h, :min_w]
            self.transmission = self.transmission[:min_h, :min_w]
    
    def _create_edge_heatmap_visualization(self, processed_image, clean_img):
        """
        Overlay dark-field image with edge information from Transmission.
        """
        if not self.params.ENABLE_EDGE_HEATMAP:
            return None

        # STEP 1: EDGES
        # ===========================================
        # Use Sobel edge detection for edge identification
        #
        # Sobel edge detection - compute both X and Y gradients for complete edge detection
        sobel_x = sobel(clean_img, axis=0)  # Vertical edges (horizontal gradient)
        sobel_y = sobel(clean_img, axis=1)  # Horizontal edges (vertical gradient)
        
        # Compute gradient magnitude to get all edges regardless of direction
        sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # STEP 2: EDGE NORMALIZATION
        # ==========================
        # NORMALIZE Sobel edges output to [0, 1]
        sobel_normalized = sobel_edges / (sobel_edges.max() + 1e-8)
        #
        # USE only Sobel edges for edge representation
        fused_edges = sobel_normalized
        
        # STEP 3: ADAPTIVE EDGE THRESHOLDING
        # ==================================
        # PERCENTILE-BASED THRESHOLDING for consistent results across images
        edge_threshold = np.percentile(fused_edges, self.params.HEATMAP_EDGE_THRESHOLD_PERCENTILE)
        edge_mask = fused_edges > edge_threshold
        #
        # APPLY smoothing to create smooth transitions
        smoothed_edges = gaussian_filter(fused_edges * edge_mask.astype(float), 
                                       sigma=self.params.HEATMAP_GAUSSIAN_SIGMA)
        # SCALE edges in the range [0, 1]
        if smoothed_edges.max() > 0:
            edge_intensity = smoothed_edges / smoothed_edges.max()
        else:
            edge_intensity = smoothed_edges
        
        # STEP 4: INTENSITY OVERLAY
        # ====================================
        # Create overlay where the dark-field image forms the base and edges 
        # appear as additive highlights. Keep processed image in its original 
        # range for information preservation.
        darkfield_base = processed_image.copy()
        #
        # CREATE edge contribution that will be added to the dark-field base
        # SCALE edges to be additive highlights based on the dark-field dynamic range
        darkfield_range = processed_image.max() - processed_image.min()
        edge_scaling_factor = darkfield_range
        #
        # CREATE edge highlights that will be added to the base image
        edge_highlights = edge_intensity * edge_scaling_factor * self.params.HEATMAP_ALPHA_BLEND
        #
        # COMBINE by adding edge highlights to the dark-field base
        # This preserves all dark-field information while adding edge information on top
        combined_image = darkfield_base + edge_highlights
        #
        # ENSURE we don't exceed bounds
        combined_max = processed_image.max() + edge_scaling_factor
        combined_image = np.clip(combined_image, processed_image.min(), combined_max)
        #
        # STORE the combined intensity data for precision saving
        self.heatmap_combined_data = combined_image.astype(np.float32)
        
        return None
    
    def _save_parameter_log(self, output_filename, used_r, used_eps, edge_similarity=None):
        """Create a parameter log file with all processing details"""
        
        # GET current date and time
        today = datetime.now()
        date_str = today.strftime(self.params.DATE_FORMAT)
        
        # CREATE base filename for log
        base_filename = f"{date_str}stairlog.txt"
        
        # FIND unique filename if file already exists
        counter = 1
        log_filename = base_filename
        while os.path.exists(log_filename):
            name, ext = os.path.splitext(base_filename)
            log_filename = f"{name}_{counter:03d}{ext}"
            counter += 1
        
        # PREPARE log content
        log_content = []
        log_content.append("=" * 80)
        log_content.append("PARAMETER LOG FOR STAIRCASE REMOVAL")
        log_content.append("=" * 80)
        log_content.append(f"Generated: {today.strftime('%Y-%m-%d %H:%M:%S')}")
        log_content.append(f"Dark-field input: {os.path.basename(self.image_path)}")
        log_content.append(f"Transmission guide: {os.path.basename(self.transmission_path)}")
        log_content.append(f"Output result: {os.path.basename(output_filename)}")
        log_content.append("")
        
        # LOG: Image information
        log_content.append("IMAGE INFORMATION:")
        log_content.append("-" * 40)
        log_content.append(f"Image dimensions: {self.dark_field.shape[0]} x {self.dark_field.shape[1]} pixels")
        log_content.append(f"Data type: {self.dark_field.dtype}")
        log_content.append(f"Dark-field intensity range: [{self.dark_field.min():.15f}, {self.dark_field.max():.15f}]")
        log_content.append(f"Transmission intensity range: [{self.transmission.min():.6f}, {self.transmission.max():.6f}]")
        log_content.append("")
        
        # LOG: Parameter optimization status
        if self.params.OPTIMIZE_BY_DEFAULT:
            log_content.append("PARAMETER OPTIMIZATION: ENABLED")
            log_content.append(f"Radius search range: {self.params.RADIUS_RANGE}")
            log_content.append(f"Regularization search range: {self.params.REGULARIZATION_RANGE}")
            if edge_similarity is not None:
                log_content.append(f"Final edge similarity score: {edge_similarity:.6f}")
        else:
            log_content.append("PARAMETER OPTIMIZATION: DISABLED")
        log_content.append("")
        
        # LOG: Guided filter parameters used
        log_content.append("GUIDED FILTER PARAMETERS USED:")
        log_content.append("-" * 40)
        log_content.append(f"Filter radius (r): {used_r}")
        log_content.append(f"Regularization (eps): {used_eps:.2e}")
        log_content.append(f"Processing window size: {2*used_r+1} x {2*used_r+1} pixels")
        log_content.append("")
        
        # LOG: Default parameters (for reference)
        log_content.append("DEFAULT PARAMETERS (for reference):")
        log_content.append("-" * 40)
        log_content.append(f"Default radius: {self.params.DEFAULT_RADIUS}")
        log_content.append(f"Default regularization: {self.params.DEFAULT_REGULARIZATION:.2e}")
        log_content.append("")
        
        # LOG: Heatmap visualization parameters
        if self.params.ENABLE_EDGE_HEATMAP:
            log_content.append("HEATMAP VISUALIZATION: ENABLED")
            log_content.append("-" * 40)
            log_content.append(f"Edge threshold percentile: {self.params.HEATMAP_EDGE_THRESHOLD_PERCENTILE}")
            log_content.append(f"Alpha blend intensity: {self.params.HEATMAP_ALPHA_BLEND}")
            log_content.append(f"Gaussian smoothing sigma: {self.params.HEATMAP_GAUSSIAN_SIGMA}")
            log_content.append("")
        else:
            log_content.append("HEATMAP VISUALIZATION: DISABLED")
            log_content.append("")
        
        # LOG: File naming parameters
        log_content.append("FILE NAMING PARAMETERS:")
        log_content.append("-" * 40)
        log_content.append(f"Date format: {self.params.DATE_FORMAT}")
        log_content.append(f"Result suffix: '{self.params.RESULT_SUFFIX}'")
        log_content.append(f"Heatmap prefix: '{self.params.HEATMAP_PREFIX}'")
        log_content.append("")
        log_content.append("")
        log_content.append("=" * 80)
        log_content.append("End of the log file")
        log_content.append("=" * 80)
        
        # WRITE log file
        try:
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_content))
            print(f"Parameter log saved as: {log_filename}")
        except Exception as e:
            print(f"Warning: Could not save parameter log: {e}")
        
        return log_filename
    
    def process(self, r=None, eps=None, optimize=None):
        """Apply guided filter and return result"""
        # USE parameters from class if not explicitly provided
        if r is None:
            r = self.params.DEFAULT_RADIUS
        if eps is None:
            eps = self.params.DEFAULT_REGULARIZATION
        if optimize is None:
            optimize = self.params.OPTIMIZE_BY_DEFAULT
        if optimize:
            optimal_r, optimal_eps = optimize_guided_filter_params(
                self.transmission, self.dark_field, 
                r_range=self.params.RADIUS_RANGE,
                eps_range=self.params.REGULARIZATION_RANGE
            )
            print(f"Using optimized parameters: r={optimal_r}, eps={optimal_eps:.2e}")
            self.clean_result = guided_filter(self.transmission, self.dark_field, 
                                            r=optimal_r, eps=optimal_eps)
            used_r, used_eps = optimal_r, optimal_eps
            # Calculate final edge similarity for logging
            final_edge_similarity = compute_edge_similarity(self.clean_result, self.transmission)
        else:
            print(f"Using default parameters: r={r}, eps={eps:.2e}")
            self.clean_result = guided_filter(self.transmission, self.dark_field, r=r, eps=eps)
            used_r, used_eps = r, eps
            final_edge_similarity = None
        
        # FILENAME from dark-field image name
        dark_field_base = os.path.splitext(os.path.basename(self.image_path))[0]
        date_str = datetime.now().strftime(self.params.DATE_FORMAT)
        
        # GET unique filename
        def get_unique_filename(base_name, extension):
            filename = f"{base_name}.{extension}"
            if not os.path.exists(filename):
                return filename
            # If file exists, append counter to create a unique filename
            counter = 1
            while True:
                filename = f"{base_name}({counter}).{extension}"
                if not os.path.exists(filename):
                    return filename
                counter += 1
        
        # SAVE main result as TIFF
        result_base = f"{dark_field_base}_{date_str}{self.params.RESULT_SUFFIX}"
        output_filename = get_unique_filename(result_base, "tif")
        
        # SAVE as float32 TIFF to preserve full data range
        imageio.imwrite(output_filename, self.clean_result.astype(np.float32))
        print(f"")
        print(f"Result saved as: {output_filename}")
        
        # SAVE multichannel heatmap visualization (ImageJ compatible)
        self._create_edge_heatmap_visualization(self.clean_result, self.transmission)
        if hasattr(self, 'heatmap_combined_data'):
            heatmap_base = f"{self.params.HEATMAP_PREFIX}{dark_field_base}_{date_str}{self.params.RESULT_SUFFIX}"
            heatmap_filename = get_unique_filename(heatmap_base, "tif")
            
            # CREATE multi-channel image for ImageJ visualization
            # Channel 1: Dark-field base (for "fire" LUT in ImageJ)
            # Channel 2: Edge overlay (for grayscale LUT in ImageJ)
            darkfield_channel = self.clean_result.astype(np.float32)
            
            # EXTRACT the edge overlay component by subtracting dark-field from combined
            edge_overlay_channel = (self.heatmap_combined_data - darkfield_channel).astype(np.float32)
            edge_overlay_channel = np.clip(edge_overlay_channel, 0, None)
            
            # STACK channels for multi-channel TIFF (ImageJ compatible)
            multi_channel_data = np.stack([darkfield_channel, edge_overlay_channel], axis=0)
            
            # SAVE as multi-channel 32-bit float TIFF
            tifffile.imwrite(heatmap_filename, multi_channel_data.astype(np.float32), 
                           imagej=True, 
                           metadata={'axes': 'CYX', 'Channel': {'Name': ['DarkField_Fire', 'EdgeOverlay_Gray']}})
            
            print(f"Multichannel heatmap saved as: {heatmap_filename}")
            #print("===================================")
            #print("SUGGESTION FOR IMAGEJ VISUALIZATION")
            #print("===================================")
            #print("   Inside ImageJ: Image > Color > Make Composite for dual-channel visualization")
            #print("   Channel 1 (DarkField): Apply 'Fire' LUT in ImageJ")
            #print("   Channel 2 (EdgeOverlay): Apply 'Yellow' LUT in ImageJ")

        # CREATE parameter log file
        self._save_parameter_log(output_filename, used_r, used_eps, final_edge_similarity)
        
        return self.clean_result

def main():
    """Main function to process actual images with guided filter."""
    try:
        processor = ImageProcessor()
        result = processor.process()
        print("")
        print("Processing complete!")
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    main()

