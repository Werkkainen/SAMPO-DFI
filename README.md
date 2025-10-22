# SAMPO-DFI
> **S**ingle-grid **A**utomated **M**odular **P**rocessing **O**verview for dark-field x-ray imaging

<br/>

Here is an overview of the demonstrative scripts for the different parts of the SAMPO-DFI pipeline. Guidance and further documentation is available inside the scripts themselves. Feel free to combine/modify the different parts into their own merged pipelines as you like. This repository includes or references external scripts that retain their original licenses. Lastly, please cite my publication if you use this repository or the findings presented in it: [LINK COMING]

<br/>

<img src="https://github.com/user-attachments/assets/34470189-a029-4aaf-88d8-872c1daf1126" width="600">


## **1) Preprocessing step**

Ensure that the reference grid-only images (_I_<sub>Ref</sub>) and the sample-plus-grid images (_I_<sub>Sample</sub>) are properly preprocessed. Your setup should be warmed up and both dark-current and flat-field calibrated. You may also wish to optionally average your image data and correct for any bad pixels in your detectors.

## **2) Single grid Fokker Planck dark-field and transmission retrieval**

The file dialog will ask you to choose the (_I_<sub>Ref</sub>) and (_I_<sub>Sample</sub>) image to retrieve the dark-field and transmission images from the data. Please set the FPParams as you like before usage.

## **3) Grid artifact removal**

The file dialog will ask you to choose:
- The image with the grid artifact
- The reference image with only the grid (_I_<sub>Ref</sub>)
- A clean flat field image without grid

High quality reference image (i.e. with averages) is recommended for the reference image with only the grid for better grid removal.

## **4) Block Matching 3D Denoising**

The file dialog will ask you to choose:
- the custom BM3D profile .py (optional) created by custom_profile_optimizer.py
- The image to denoise

The custom denoiser can be created with your setups data by 4)_custom_profile_optimizer.py (see the brief description inside the script documentation) or you can use a default denoising profiles.

## **5) Staircase suppression**

The file dialog will ask you to choose:
- Dark-field image with staircase artifacts
- Transmission image for guidance

This will give you the processed chosen image with staircase artifacts suppressed. Extra heatmap visualization saved as multichannel tiff for ImageJ compatibility. For instance, inside ImageJ:
  - Image > Color > Make Composite for dual-channel visualization
  - Channel 1 (DarkField): Apply 'Fire' LUT in ImageJ
  - Channel 2 (EdgeOverlay): Apply 'Yellow' LUT in ImageJ

<br/>

> [!IMPORTANT]  
> Please check the requirements.txt file for the complete list of dependencies that were present and their versions.<br>
> **Python 3.12.0** <br>
> opencv 4.11.0.86<br>
> matplotlib 3.10.3<br>
> numpy 2.3.1<br>
> pillow  11.3.0<br>
> scipy  1.16.0<br>
> scikit-image 0.25.2
