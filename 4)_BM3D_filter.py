# USE all CPU cores for parallel processing
import os as _os
_workers = (_os.cpu_count() or 1)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ.setdefault(_var, str(_workers))
try:
    import numba as _nb
    try:
        _nb.set_num_threads(_workers)
    except Exception:
        pass
except Exception:
    pass
import numpy as np
from PIL import Image
from bm3d import bm3d, BM3DProfile
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog
import time
import importlib.util

########################################
#                                      #
#  Block Matching 3D (BM3D) Denoising  #
#                                      # 
########################################

# ==================================================================================
# DESCRIPTION:
# > Select via file dialog: 
#   - the custom BM3D profile .py (optional) created by custom_profile_optimizer.py
#   - The image to denoise (TIFF format, grayscale)
# > Results are saved in the same folder as the input images.
# > Log file is saved with all parameters.
# ==================================================================================

"""
The scripts BM3D library is based on:

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

# IMPORT the optimized custom BM3D profile via a file selection dialog (optional)
CUSTOM_OPTIMAL_PSD = None  # Optional PSD value provided by the custom profile module

def _load_custom_profile_via_dialog():
    """Let the user pick a Python module that defines create_optimized_streak_profile().
    Returns (module, factory_func) or (None, None) if not selected/invalid.
    """
    try:
        Tk().withdraw()
        selected_file = filedialog.askopenfilename(
            title="Select custom BM3D profile module (optional)",
            filetypes=[("Python files", "*.py"), ("All Files", "*.*")]
        )
        if not selected_file:
            return None, None
        spec = importlib.util.spec_from_file_location("optimized_profile", selected_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factory = getattr(module, "create_optimized_streak_profile", None)
        if factory is None:
            print("Selected module does not define 'create_optimized_streak_profile'.")
            return None, None

        # TRY to obtain an optimal PSD value from the module (optional)
        optimal_psd = None
        for name in ("OPTIMAL_PSD", "OPTIMIZED_PSD", "optimal_psd", "optimized_psd"):
            if hasattr(module, name):
                try:
                    optimal_psd = float(getattr(module, name))
                    break
                except Exception:
                    pass
        if optimal_psd is None:
            for getter in ("get_optimized_psd", "get_optimal_psd"):
                if hasattr(module, getter):
                    try:
                        optimal_psd = float(getattr(module, getter)())
                        break
                    except Exception:
                        pass

        print(f"Custom optimized profile loaded from: {os.path.basename(selected_file)}" + (f" (PSD={optimal_psd})" if optimal_psd is not None else ""))
        return module, factory, optimal_psd
    except Exception as e:
        print(f"Failed to load custom profile: {e}")
        return None, None, None

# TRY to load a custom profile interactively; fall back to standard profiles if skipped/invalid
try:
    optimized_profile_module, _create_factory, _optimal_psd = _load_custom_profile_via_dialog()
    if _create_factory is None:
        raise ImportError("No custom profile selected or missing factory function")
    create_optimized_streak_profile = _create_factory
    CUSTOM_OPTIMAL_PSD = _optimal_psd
    CUSTOM_PROFILE_AVAILABLE = True
except Exception as e:
    print(f"Custom profile not selected/loaded: {e}")
    print("  Using standard BM3D profiles instead.")
    CUSTOM_PROFILE_AVAILABLE = False


# ============= PARAMETERS ===================================================================
class ProcessingParams:

    def __init__(self):
        self.passes = 2              # Number of BM3D passes for final denoising
        self.psd_range = (0.1, 2.0)  # Fallback range for default PSD selection
        # No optimization in this script: a fixed PSD is used (from custom profile if provided)
        # BM3D profiles: 'np' (normal, balanced), 'refilter', 'vn', 'high' (best quality), 'vn_old', 'deb'
        # 'np' is often better for structured artifacts like streaks
        # BM3D profile for streak removal:
        # 'np'       - Normal: Balanced, good for most streak patterns (DEFAULT)
        # 'vn'       - Very aggressive noise reduction, good for high noise
        # 'high'     - High Quality: Slowest, best for complex/fine streaks
        # 'refilter' - Refiltering mode for enhanced quality
        # 'vn_old'   - Legacy very aggressive mode
        # 'deb'      - Deblocking mode
        # 'custom'   - Use optimized custom profile for streak removal (when available)
        self.bm3d_profile = 'custom' if CUSTOM_PROFILE_AVAILABLE else 'np'
        
        # Custom profile instance (will be created when needed)
        self.custom_profile_instance = None
        
        # Fixed PSD to use when custom profile doesn't supply one (midpoint of range by default)
        self.fixed_psd = (self.psd_range[0] + self.psd_range[1]) / 2.0
    
    def get_bm3d_profile(self):
        """Get the appropriate BM3D profile for denoising"""
        if self.bm3d_profile == 'custom' and CUSTOM_PROFILE_AVAILABLE:
            if self.custom_profile_instance is None:
                try:
                    self.custom_profile_instance = create_optimized_streak_profile()
                except AttributeError as e:
                    print(f"Custom profile incompatible: {e}")
                    print("Falling back to standard profile 'np'.")
                    self.bm3d_profile = 'np'
                    return self.bm3d_profile
            return self.custom_profile_instance
        else:
            return self.bm3d_profile

    def get_effective_psd(self) -> float:
        """Return the PSD value to use for denoising.
        Prefer a PSD provided by the custom profile module; otherwise use fixed_psd.
        """
        if CUSTOM_OPTIMAL_PSD is not None and np.isfinite(CUSTOM_OPTIMAL_PSD) and CUSTOM_OPTIMAL_PSD > 0:
            return float(CUSTOM_OPTIMAL_PSD)
        return float(self.fixed_psd)

# ============= THE DENOISE CLASS =======================================================
class Denoise:
    def __init__(self, params: ProcessingParams):
        self.params = params
        self.img = None
        self.img_np = None
        self.img_min = None
        self.img_max = None
        self.denoised = None
        self.optimized_denoised = None
        self.optimal_psd = None

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

    def bm3d_denoise(self):
        file_path = self.select_image("Select the image to denoise")
        self.img, self.img_np = self.load_image(file_path)
        self.img_min, self.img_max = float(self.img_np.min()), float(self.img_np.max())
        img_norm = (self.img_np - self.img_min) / max(self.img_max - self.img_min, 1e-8)

        # DETERMINE PSD and profile
        psd_value = self.params.get_effective_psd()
        profile = self.params.get_bm3d_profile()
        print(f"Using profile: {profile if isinstance(profile, str) else 'custom_profile'} | PSD={psd_value} | passes={self.params.passes}")

        # APPLY BM3D passes
        current = np.ascontiguousarray(img_norm.astype(np.float32, copy=False))
        for i in range(int(self.params.passes)):
            try:
                current = bm3d(current, psd_value, profile=profile)
            except OSError as e:
                print(f"bm3d crashed with profile {profile} (pass {i+1}): {e}. Falling back to 'np'.")
                current = bm3d(current, psd_value, profile='np')

        # BACK to original scale and save
        denoised = current * (self.img_max - self.img_min) + self.img_min
        denoised = np.clip(denoised, self.img_min, self.img_max)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}_BM3Dc.tif"
        self.save_image(denoised, self.img, self.img_np, output_filename)
        print(f"Saved: {output_filename}")
        self.display_images([self.img_np, denoised], ["Original", "BM3D Denoised"], (12, 5))

    def save_image(self, arr, img, img_np, filename):
        # SAVE with the same dtype as the original image
        out_img = Image.fromarray(arr.astype(img_np.dtype))
        out_img.save(filename)

    def denoise_with_psd(self, img_norm, psd_value, roi_only=False):
        """Apply BM3D denoising with specific PSD value"""
        try:
            psd_value = float(psd_value)
        except Exception:
            raise ValueError(f"Invalid PSD value: {psd_value}")
        if not np.isfinite(psd_value) or psd_value <= 0:
            raise ValueError(f"PSD must be finite and > 0, got {psd_value}")

        # ENSURE dtype and memory layout are compatible
        img_norm = np.ascontiguousarray(img_norm.astype(np.float32, copy=False))

        current_norm = img_norm.copy()
        for i in range(self.params.passes):
            try:
                current_norm = bm3d(current_norm, psd_value, profile=self.params.get_bm3d_profile())
            except OSError as e:
                print(f"bm3d crashed with custom profile (full pass {i+1}): {e}")
                print("Retrying with standard profile 'np'.")
                current_norm = bm3d(current_norm, psd_value, profile='np')
        return current_norm


    def bm3d_denoise_optimized(self):
        """Denoise using the predefined/custom profile only (no optimization)."""
        file_path = self.select_image("Select the image to denoise")
        self.file_path = file_path
        self.img, self.img_np = self.load_image(file_path)
        self.img_min, self.img_max = float(self.img_np.min()), float(self.img_np.max())
        self.img_norm = (self.img_np - self.img_min) / max(self.img_max - self.img_min, 1e-8)
        profile = self.params.get_bm3d_profile()
        self.optimal_psd = self.params.get_effective_psd()

        print(f"Running...")

        start_time = time.time()
        current_norm = np.ascontiguousarray(self.img_norm.astype(np.float32, copy=False))
        for i in range(int(self.params.passes)):
            try:
                current_norm = bm3d(current_norm, self.optimal_psd, profile=profile)
            except OSError as e:
                print(f"bm3d crashed with profile {profile} (pass {i+1}): {e}. Retrying with 'np'.")
                current_norm = bm3d(current_norm, self.optimal_psd, profile='np')
        elapsed = time.time() - start_time
        print(f"BM3D done in {elapsed:.2f}s")

        self.optimized_denoised = current_norm
        self.optimized_denoised_original_scale = np.clip(
            self.optimized_denoised * (self.img_max - self.img_min) + self.img_min,
            self.img_min,
            self.img_max,
        )

        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        output_filename = f"{base_name}_BM3Dc.tif"
        self.save_image(self.optimized_denoised_original_scale, self.img, self.img_np, output_filename)
        print(f"Saved: {output_filename}")

        self.display_images([
            self.img_np,
            self.optimized_denoised_original_scale,
        ], [
            'Original Image',
            f'BM3D (PSD={self.optimal_psd:.3f})',
        ], figsize=(12, 5))

    def display_images(self, images, titles, figsize=(15, 5)):
        """Display multiple images side by side"""
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        if len(images) == 1:
            axes = [axes]
        
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


# ============== MAIN EXECUTION ==============================================================

def main():
    params = ProcessingParams()
    denoiser = Denoise(params)
    
    print("")
    print("Custom BM3D Denoising")
    print("=====================")
    print(f"Threads: {_workers}")
    print(f"PSD: {params.get_effective_psd()}")
    print(f"BM3D passes: {params.passes}")
    print(f"BM3D profile: {params.bm3d_profile}")
    print()

    denoiser.bm3d_denoise_optimized()

if __name__ == "__main__":
    main()