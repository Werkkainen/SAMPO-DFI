# SAMPO-DFI
**S**ingle-grid **A**utomated **M**odular **P**rocessing **O**verview for dark-field x-ray imaging


Here is an overview of the demonstrative scripts for the different parts of the SAMPO-DFI pipeline. Guidance and further documentation is available inside the scripts themselves.

INPUTS:  -Reference grid-only x-ray image, I_Ref
         -Sample+grid image x-ray image, I_Sample


**1) Preprocessing**
Make sure the reference grid-only images I_Ref and sample+grid image I_Sample data are preprocessed. So your setup is well dark-current and flat-field calibrated, warmed up before use, you may want to do optional averaging of your image data, and bad-pixels of your detectors are accounted for.

**2) Single grid Fokker Planck dark-field and transmission retrieval**
The file dialog will ask you to choose the I_Ref and I_Sample image to retrieve the dark-field and transmission images from the data. Please set the FPParams as you like before usage.

**3) Grid artifact removal**
The file dialog will ask you to choose:
 	-The image with the grid artifacts
 	-The reference image with only the grid
 	-A clean flat field image without grid
High quality reference image (i.e. with averages) is recommended for the reference image with only the grid for better grid removal.

**4) Block Matching 3D Denoising**
The file dialog will ask you to choose:
 	-the custom BM3D profile .py (optional) created by custom_profile_optimizer.py
 	-The image to denoise
The custom denoiser can be created with your setups data by 4)_custom_profile_optimizer.py (see the brief description inside the script documentation) or you can use the default denoising profiles.

**5) Staircase removal**
The file dialog will ask you to choose:
 	-Dark-field image with staircase artifacts
 	-Transmission image for guidance
This will give you the processed chosen image with staircase artifacts removed.
+Extra heatmap visualization saved as multichannel tiff for ImageJ compatibility. For instance, inside ImageJ:
 	Image > Color > Make Composite for dual-channel visualization
 	Channel 1 (DarkField): Apply 'Fire' LUT in ImageJ
 	Channel 2 (EdgeOverlay): Apply 'Yellow' LUT in ImageJ


**Final notes**
Feel free to combine/modify the different parts into their own merged pipelines as you like. References and detailed documentation is found inside the scripts themselves. This repository includes or references external scripts that retain their original licenses.
