# ---------------------------------------------------------------------------------
# Base code written by Samantha Alloo // Date 23.10.2024
# Original source: amanthaalloo (2024), https://github.com/samanthaalloo/EvVSDev_SBXIFokkerPlanck
# Alloo, S. J., Paganin, D. M., Croughan, M. K., Ahlers, J. N., Pavlov, K. M. &
# Morgan, K. S. (2025), ‘Separating edges from microstructure in x-ray dark-field
# imaging: evolving and devolving perspectives via the x-ray fokker-planck equation’,
# Opt. Express 33(2), 3577–3600.
# 
# Slight modifications for clarity and structure. Furthermore, the variable C
# implemented for spatial frequency cutoff. Some parts of the script were written with 
# assistance from GitHub Copilot.
# ---------------------------------------------------------------------------------
from decimal import Decimal, getcontext
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math
import scipy
from scipy import ndimage, misc
from PIL import Image
import time
from scipy.ndimage import median_filter, gaussian_filter
import fabio
import pyedflib
import h5py
from tkinter import Tk, filedialog
from dataclasses import dataclass, field

# ==================================================================================
# DESCRIPTION:
# > Select two images via file dialog: one mask-only image and one sample-plus-mask image.
# > Dark-field diffusion coefficient D and transmission T are retrieved (evolving and devolving)
# > Log file is saved with all parameters.
# ==================================================================================


# ---------------------------------------------------------------------------------
getcontext().prec = 50  # Decimal precision

@dataclass
class FPParams:
    """All configurable parameters used by the FP retrieval.

    Values are in microns [um] where applicable. Derived quantities like
    effective propagation distance, pixel size, Nyquist frequency, and C are
    exposed as properties.
    """

    # Primary settings
    M: float = 2000 / 115                 # Magnification
    prop_numerator_um: float = 0.74e6     # numerator for effective propagation (divided by M)
    pixel_size_detector_um: float = 50    # detector pixel size before magnification [um]
    pixel_size_cutoff_um: float = 50      # cutoff pixel size used for Nyquist/C [um]
    f: float = 0.9                        # spatial frequency cutoff (dimensionless)
    #p: float = 10                        # speckle period for visibility reduction calculation
    savedir: str = field(default_factory=os.getcwd)

    # UI and I/O strings
    ui_prompt_only_mask: str = "Select the only mask image"
    ui_prompt_with_object: str = "Select the image with the object"
    ui_filetypes: tuple = (("All Files", "*.*"),)
    out_tie_transmission_single_evolving: str = "TIE_T_evolve.tif"
    out_tie_transmission_single_devolving: str = "TIE_T_devolve.tif"
    out_D_evolve_fmt: str = "FP_D_evolve_f{f:.1f}.tif"
    out_D_devolve_fmt: str = "FP_D_devolve_f{f:.1f}.tif"
    out_V_evo_fmt: str = "V_Evo_p{p}.tif"
    out_V_devo_fmt: str = "V_Devo_p{p}.tif"

    # Numerical thresholds and regularization
    tie_min_transmission_threshold: float = 1e-10
    inv_laplacian_reg: float = 1e-4

    @property
    def prop(self) -> float:
        """Effective propagation distance [um]."""
        return self.prop_numerator_um / self.M

    @property
    def pixel_size(self) -> float:
        """Effective pixel size at sample plane [um]."""
        return self.pixel_size_detector_um / self.M

    @property
    def nyquist(self) -> Decimal:
        """Nyquist angular spatial frequency [1/um] as Decimal."""
        eff_cutoff = self.pixel_size_cutoff_um / self.M
        value = 2 * math.pi / (2 * eff_cutoff)
        return Decimal(str(value))

    @property
    def C(self) -> Decimal:
        """Regularization constant C (Decimal)."""
        f_dec = self.f if isinstance(self.f, Decimal) else Decimal(str(self.f))
        Ny = self.nyquist
        return Decimal(1) / (f_dec ** 2 * Ny ** 2)



# ---------------------------------------------------------------------------------
# Defining additional functions
def preprocess_array(arr):
    # Ensure it's a numpy array
    arr = np.array(arr)
    # Remove extra dimensions
    arr = np.squeeze(arr)
    # Ensure it's float64
    arr = arr.astype(np.float64)
    # Replace NaNs and Infs (safety step even if none are present)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

# ---------------------------------------------------------------------------------
# Output helpers: datestamp and unique filename generation
def _datestamped_name(base_filename: str, date_str: str) -> str:
    name, ext = os.path.splitext(base_filename)
    return f"{name}_{date_str}{ext}"

def build_unique_savepath(savedir: str, base_filename: str) -> str:
    """Return a full path with YYYYMMDD datestamp and (1),(2),… if needed.

    - base_filename: e.g., "FP_D_evolve_f0.2.tif"
    - result: e.g., "FP_D_evolve_f0.2_20250815.tif" or "FP_D_evolve_f0.2_20250815 (1).tif"
    """
    date_str = datetime.now().strftime("%Y%m%d")
    stamped = _datestamped_name(base_filename, date_str)
    full = os.path.join(savedir, stamped)
    if not os.path.exists(full):
        return full
    name, ext = os.path.splitext(stamped)
    idx = 1
    while True:
        candidate = os.path.join(savedir, f"{name}_({idx}){ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1

def write_run_log(params: FPParams, inputs: dict, outputs: dict, savedir: str) -> str:
    """Write a log with inputs, outputs, parameters, and derived properties.

    Returns the path to the written log file.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = build_unique_savepath(savedir, "FP_RunLog.txt")

    def _fmt_decimal(d):
        try:
            return str(d)
        except Exception:
            return repr(d)

    lines = []
    lines.append("FP Single-Shot Run Log")
    lines.append(f"Date/Time: {now}")
    lines.append("")
    lines.append("Inputs:")
    for k, v in inputs.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Outputs:")
    for k, v in outputs.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Primary settings:")
    lines.append(f"  M: {params.M}")
    lines.append(f"  prop_numerator_um: {params.prop_numerator_um}")
    lines.append(f"  pixel_size_detector_um: {params.pixel_size_detector_um}")
    lines.append(f"  pixel_size_cutoff_um: {params.pixel_size_cutoff_um}")
    lines.append(f"  f: {params.f}")
    lines.append(f"  savedir: {params.savedir}")
    lines.append("")
    lines.append("Numerical thresholds and regularization:")
    lines.append(f"  tie_min_transmission_threshold: {params.tie_min_transmission_threshold}")
    lines.append(f"  inv_laplacian_reg: {params.inv_laplacian_reg}")
    lines.append("")
    lines.append("Derived properties:")
    lines.append(f"  prop [um]: {params.prop}")
    lines.append(f"  pixel_size [um]: {params.pixel_size}")
    lines.append(f"  nyquist [1/um]: {_fmt_decimal(params.nyquist)}")
    lines.append(f"  C: {_fmt_decimal(params.C)}")

    content = "\n".join(lines) + "\n"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content)
    return log_path

def kspace_kykx(image_shape: tuple, pixel_size: float = 1):
    # Multiply by 2pi for correct values, since DFT has 2pi in exponent
    rows = image_shape[0]
    columns = image_shape[1]
    ky = 2*math.pi*scipy.fft.fftfreq(rows, d=pixel_size) # spatial frequencies relating to "rows" in real space
    kx = 2*math.pi*scipy.fft.fftfreq(columns, d=pixel_size) # spatial frequencies relating to "columns" in real space
    return ky, kx

def invLaplacian(image, pixel_size, regkr2=1e-4):
    # Need to mirror the image to enforce periodicity
    flip = np.concatenate((image, np.flipud(image)), axis=0)
    flip = np.concatenate((flip, np.fliplr(flip)), axis=1)

    ky, kx = kspace_kykx(flip.shape,pixel_size)
    ky2 = ky**2
    kx2 = kx**2

    kr2 = np.add.outer(ky2, kx2)
    ftimage = np.fft.fft2(flip)
    regdiv = 1/(kr2+regkr2)
    invlapimageflip = -1*np.fft.ifft2(regdiv*ftimage)

    row = int(image.shape[0])
    column = int(image.shape[1])

    invlap = np.real(invlapimageflip[0:row,0:column])
    return invlap, regkr2

def xderivative(image,pixel_size):
    im_mirror_h = np.concatenate((image, np.fliplr(image)), axis=1) # Doing mirroring (horizontally and vertically) to enforce periodicity
    im_mirror_v = np.concatenate((im_mirror_h, np.flipud(im_mirror_h)), axis=0)

    ky, kx = kspace_kykx(im_mirror_v.shape, pixel_size)

    ky0 = ky * 0
    i_kx = kx * (np.zeros((kx.shape),
                          dtype=np.complex128) + 0 + 1j)  # (i) * derivative along rows of DF, has "0" in the real components, and "d(DF)/dx" in the complex

    i_kx_0ky = np.add.outer(ky0, i_kx)

    fft_im = scipy.fft.fft2(im_mirror_v)
    kernfft_dx = i_kx_0ky * fft_im
    dx_im = np.real(scipy.fft.ifft2(kernfft_dx))

    dx_im_crop = dx_im[:image.shape[0],:image.shape[1]]

    return dx_im_crop

def yderivative(image,pixel_size):
    im_mirror_h = np.concatenate((image, np.fliplr(image)), axis=1) # Doing mirroring (horizontally and vertically) to enforce periodicity
    im_mirror_v = np.concatenate((im_mirror_h, np.flipud(im_mirror_h)), axis=0)

    ky, kx = kspace_kykx(im_mirror_v.shape, pixel_size)

    kx0 = kx * 0
    i_ky = ky * (np.zeros((ky.shape),
                          dtype=np.complex128) + 0 + 1j)  # (i) * derivative along rows of DF, has "0" in the real components, and "d(DF)/dx" in the complex

    i_ky_0kx = np.add.outer(i_ky, kx0)

    fft_im = scipy.fft.fft2(im_mirror_v)
    kernfft_dy = i_ky_0kx * fft_im
    dy_im = np.real(scipy.fft.ifft2(kernfft_dy))

    dy_im_crop = dy_im[:image.shape[0], :image.shape[1]]
    return dy_im_crop

def lowpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a low-pass filter and suppress the input spatial frequencies, kr of the image, beyond some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    lowpass_2d = np.exp(-r * (kr ** 2))

    # plt.imshow(lowpass_2d)
    # plt.title('Low-Pass Filter 2D')
    # plt.colorbar()
    # plt.show()

    return lowpass_2d

def highpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a high-pass filter and suppress the input spatial frequencies, kr of the image, up to some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    highpass_2d = 1 - np.exp(-r * (kr ** 2))

    # plt.imshow(highpass_2d)
    # plt.title('High-Pass Filter 2D')
    # plt.colorbar()
    # plt.show()

    return highpass_2d

def midpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a low-pass filter and suppress the input spatial frequencies, kr of the image, up to some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    highpass_2d = 1 - np.exp(-r * (kr ** 2))

    C = np.zeros(columns, dtype=np.complex128)
    C = C + 0 + 1j
    ikx = kx * C  # (i) * spatial frequencies in x direction (along columns) - as complex numbers ( has "0" in the real components, and "kx" in the complex)
    denom = np.add.outer((-1 * ky), ikx)  # array with ikx - ky (DENOMINATOR)

    midpass_2d = np.divide(complex(1., 0.) * highpass_2d, denom, out=np.zeros_like(complex(1., 0.) * highpass_2d),
                           where=denom != 0)  # Setting output equal to zero where denominator equals zero

    # plt.imshow(np.real(midpass_2d))
    # plt.title('Mid-Pass Filter 2D')
    # plt.colorbar()
    # plt.show()

    return midpass_2d
# ---------------------------------------------------------------------------------
# Here are all of the solutions for the different inverse problems
# 1) Transport-of-intensity equation single-exposure speckle-based X-ray imaging phase-retrieval algorithm:

def TIE_Speckle(Is, Ir, params):
    # ---------------------------------------------------------------
    # Implementing the approach in:
    # K. M. Pavlov, H. Li, D. M. Paganin, et al., “Single-shot x-ray speckle-based imaging of a single-material object,”
    # Phys. Rev. Appl. 13, 054023 (2020).
    # ---------------------------------------------------------------
    # Definitions:
    # Is: One sample-plus-speckle image [ndarray]
    # Ir: One speckle-only image [ndarray]
    # params: FPParams containing pixel_size [um], C, prop [um]
    # ---------------------------------------------------------------

    # Safety clamp to avoid division by zero in ratio Is/Ir
    eps = np.finfo(np.float64).eps
    safe_Ir = np.maximum(Ir, eps)
    IsIr = Is / safe_Ir

    IsIr_mirror = np.concatenate((IsIr, np.fliplr(IsIr)), axis=1) # Doing mirroring (horizontally and vertically) to enforce periodicity for DFT implementation
    IsIr_mirror = np.concatenate((IsIr_mirror, np.flipud(IsIr_mirror)), axis=0)

    ft_IsIr = np.fft.fft2(IsIr_mirror) # Taking the 2D Fourier Transform
    ky, kx = kspace_kykx(ft_IsIr.shape, params.pixel_size) # Finding the Fourier-space spatial frequencies
    ky2kx2 = np.add.outer(ky**2,kx**2) # Making the k_x^2 + k_y^2 term with correct dimensions
    C_float = float(params.C)
    ins_ifft = ft_IsIr / (1 + C_float * ky2kx2) # Change here
    Iob = np.real(np.fft.ifft2(ins_ifft)) # Taking the inverse Fourier transform and only look at the real component to give the transmission term
    
    # Apply thresholding to remove negative values and outliers
    threshold = params.tie_min_transmission_threshold  # Small positive value close to zero
    Iob[Iob < threshold] = threshold
    
    Iob_crop = Iob[:Is.shape[0], :Is.shape[1]] # Cropping off the mirror done

    # Safety note: phase retrieval removed; ensure positive transmission for any downstream logs
    if np.any(Iob_crop <= 0):
        print("TIE transmission had <= 0 values after thresholding; clamped to threshold.")

    return Iob_crop, None

# 2) Single-exposure evolving speckle-based X-ray imaging Fokker--Planck perspective

def Single_Evolving(Is, Ir, params):
    # ---------------------------------------------------------------
    # Definitions:
    # Is: One sample-plus-speckle image [ndarray]
    # Ir: One speckle-only image [ndarray]
    # params: FPParams containing pixel_size [um], C, prop [um], savedir, etc.
    # ---------------------------------------------------------------
    # Step 1: Approximate the sample's transmission term using TIE Speckle method 
    C = float(params.C)
    transmission, phase = TIE_Speckle(Is, Ir, params)
    tie_evo_path = build_unique_savepath(params.savedir, params.out_tie_transmission_single_evolving)
    Image.fromarray(transmission.astype(np.float32)).save(tie_evo_path)

    # Modify the speckle-only image by multiplying it with the transmission
    Ir_Tran = Ir * transmission

    # Calculate the flux difference between modified speckle image and sample-plus-speckle image
    Flux = Ir_Tran - Is

    # Check for invalid values before taking the log
    if np.any(transmission <= 0):
        print("Invalid values in transmission:", transmission[transmission <= 0])

    # Compute the logarithmic transmission to determine the optical flow field
    ln_trans = np.log(transmission)

    # Step 2: Calculate the optical flow using the derivative of the transmission
    # The flow accounts for phase changes due to sample interaction with the X-rays
   
    Flow = C * (
            xderivative(Ir_Tran * xderivative(ln_trans, params.pixel_size), params.pixel_size) +
            yderivative(Ir_Tran * yderivative(ln_trans, params.pixel_size), params.pixel_size))

    # Subtract the flux from the calculated flow to obtain the final Flow minus Flux term
    FlowMINUSFlux = Flow - Flux

    # Step 3: Invert the Laplacian of the Flow minus Flux term to obtain a smoother solution
    invlapFF, reg = invLaplacian(FlowMINUSFlux, params.pixel_size, params.inv_laplacian_reg)

    # Step 4: Compute the diffusion coefficient D using the inverted Laplacian
    prop = float(params.prop)
    D = invlapFF / (prop ** 2 * Ir_Tran)

    # Separate D into positive and negative values for further analysis
    positive_D = np.clip(D, 0, np.inf)  # Clipping at 0 to get positive values only
    negative_D = np.clip(D, -np.inf, 0)  # Clipping at 0 to get negative values only

    print('Single-exposure evolving SBXI Fokker-Planck inverse problem has been solved!')
    # Return the diffusion coefficient and its positive/negative components along with the transmission term
    return D, positive_D, negative_D, transmission, tie_evo_path

# 3) Single-exposure devolving speckle-based X-ray imaging Fokker--Planck perspective

def Single_Devolving(Is, Ir, params):
    # ---------------------------------------------------------------
    # Implementing the approach in:
    # M. A. Beltran, D. M. Paganin, M. K. Croughan, and K. S. Morgan, “Fast implicit diffusive dark-field retrieval for
    # single-exposure, single-mask x-ray imaging,” Optica 10, 422–429 (2023).
    # ---------------------------------------------------------------
    # Definitions:
    # Is: One sample-plus-speckle image [ndarray]
    # Ir: One speckle-only image [ndarray]
    # params: FPParams containing pixel_size [um], C, prop [um], savedir, etc.
    # ---------------------------------------------------------------

    # Step 1: Approximate the sample's transmission and phase using TIE Speckle method
    C = float(params.C)
    transmission, phase = TIE_Speckle(Is, Ir, params)
    # Save the TIE transmission from the devolving path as well
    tie_devo_path = build_unique_savepath(params.savedir, params.out_tie_transmission_single_devolving)
    Image.fromarray(transmission.astype(np.float32)).save(tie_devo_path)

    # Step 2: Calculate the flux, which represents the difference between the sample-plus-speckle
    # image and the modified speckle-only image (scaled by transmission)
    Flux = Is - Ir * transmission

    # Check for invalid values before taking the log
    if np.any(transmission <= 0):
        print("Invalid values in transmission:", transmission[transmission <= 0])

    # Compute the logarithmic transmission to derive the optical flow
    ln_trans = np.log(transmission)

    # Step 3: Calculate the optical flow based on the derivatives of the transmission
    # The flow characterizes how the phase changes across the image
    Flow = C * (
            xderivative(Is * xderivative(ln_trans, params.pixel_size), params.pixel_size) +
            yderivative(Is * yderivative(ln_trans, params.pixel_size), params.pixel_size))

    # Add the flux and flow to generate the final term used in the Laplacian inversion
    FluxaddFlow = Flux + Flow

    # Step 4: Invert the Laplacian of the Flux plus Flow term to obtain a smoother solution
    invlapFF, reg = invLaplacian(FluxaddFlow, params.pixel_size, params.inv_laplacian_reg)

    # Step 5: Compute the diffusion coefficient D from the inverted Laplacian
    prop = float(params.prop)
    D = invlapFF / (prop ** 2 * Is)

    # Separate D into positive and negative values for further analysis
    positive_D = np.clip(D, 0, np.inf)  # Positive values of D
    negative_D = np.clip(D, -np.inf, 0)  # Negative values of D

    # Step 6: Save the transmission and diffusion coefficient images
    #os.chdir(savedir)  # Change directory to the save location
    #Image.fromarray(transmission).save('TIE_Transmission.tif')  # Save transmission image
    #Image.fromarray(D).save('XDF_SingleDevolve_{}.tif'.format(str(reg)))  # Save diffusion image
    #Image.fromarray(positive_D).save('Pos_SingleDevolve_{}.tif'.format(str(reg)))  # Save positive diffusion image
    #Image.fromarray(-1 * negative_D).save(
     #   'Neg_SingleDevolve_{}.tif'.format(str(reg)))  # Save negative diffusion image (inverted)
    print('Single-exposure devolving SBXI Fokker-Planck inverse problem has been solved!')
    # Return the diffusion coefficient and its positive/negative components along with the transmission term
    return D, positive_D, negative_D, transmission, tie_devo_path

# Plotting function
def plot_results(images, titles, cmap='gray'):
    plt.figure(figsize=(15, 10))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.colorbar()
    plt.show()

# -------------------------------------------------------------------
# Running the functions

def main():
    # Initialize parameters
    params = FPParams()
    # Ensure save directory exists
    os.makedirs(params.savedir, exist_ok=True)

    # OPEN file selection dialogs using parameterized prompts
    print(params.ui_prompt_only_mask)
    file_path_OnlyMask = filedialog.askopenfilename(title=params.ui_prompt_only_mask, filetypes=list(params.ui_filetypes))
    if not file_path_OnlyMask or not os.path.isfile(file_path_OnlyMask):
        raise FileNotFoundError(f"First file not found or selection canceled: {file_path_OnlyMask}")

    print(params.ui_prompt_with_object)
    file_path_RawData = filedialog.askopenfilename(title=params.ui_prompt_with_object, filetypes=list(params.ui_filetypes))
    if not file_path_RawData or not os.path.isfile(file_path_RawData):
        raise FileNotFoundError(f"Second file not found or selection canceled: {file_path_RawData}")

    # READ and check images
    data_OnlyMask = np.array(Image.open(file_path_OnlyMask)).astype(float)
    data_Sample = np.array(Image.open(file_path_RawData)).astype(float)
    if len(data_OnlyMask.shape) > 2:
        raise ValueError("The first image is not grayscale.")
    if len(data_Sample.shape) > 2:
        raise ValueError("The second image is not grayscale.")

    # Preprocess
    data_Sample = preprocess_array(data_Sample)
    data_OnlyMask = preprocess_array(data_OnlyMask)

    # Run Single_Evolving and Single_Devolving with current parameters
    D_Sev, positive_D_Sev, negative_D_Sev, transmission, tie_evo_path = Single_Evolving(data_Sample, data_OnlyMask, params)
    D_Sdev, positive_D_Sdev, negative_D_Sdev, transmission, tie_devo_path = Single_Devolving(data_Sample, data_OnlyMask, params)

    # CALCULATE visibility reduction
    #V_Evo = np.exp(-(4 * np.pi * D_Sev * params.prop ** 2) / params.p ** 2)
    #V_Devo = np.exp(-(4 * np.pi * D_Sdev * params.prop ** 2) / params.p ** 2)

    # Save the results using parameterized names
    D_evo_path = build_unique_savepath(params.savedir, params.out_D_evolve_fmt.format(f=float(params.f)))
    D_devo_path = build_unique_savepath(params.savedir, params.out_D_devolve_fmt.format(f=float(params.f)))
    Image.fromarray(D_Sev.astype(np.float32)).save(D_evo_path)
    Image.fromarray(D_Sdev.astype(np.float32)).save(D_devo_path)
    
    # Write run log
    inputs = {
        "OnlyMask": file_path_OnlyMask,
        "WithObject": file_path_RawData,
    }
    outputs = {
        "TIE_Transmission_Evolving": tie_evo_path,
        "TIE_Transmission_Devolving": tie_devo_path,
        "D_Evolve": D_evo_path,
        "D_Devolve": D_devo_path,
    }
    log_path = write_run_log(params, inputs, outputs, params.savedir)
    print(f"Run log saved: {log_path}")
    #Image.fromarray(V_Evo).save(os.path.join(params.savedir, params.out_V_evo_fmt.format(p=params.p)))
    #Image.fromarray(V_Devo).save(os.path.join(params.savedir, params.out_V_devo_fmt.format(p=params.p)))

if __name__ == "__main__":
    main()
