# Wound Healing Assay Dataset (WHAD) and Cell Adhesion and Motility Assay Dataset (CAMAD)

This repository contains the codes for visualizing and processing the Wound Healing Assay Dataset (WHAD) and Cell Adhesion and Motility Assay Dataset (CAMAD) datasets. The datasets are available at XXXX.

The WHAD dataset contains images of MCF7 and MCF10A cells in 2D wound healing assays. The CAMAD dataset contains images and videos of MCF7 cells in 3D cell adhesion and motility assays.

The directory structure for the datasets is as follows:
## WHAD Directory Structure
```
..\data\whad
|-- images
| |-- MCF7
| | |-- ... (assay subfolders)
| | | |-- assay_name_frame_001.tif
| | | -- ...
| |-- MCF10A\
| | |-- ... (assay subfolders) \
| | | |-- assay_name_frame_001.tif
| | | -- ...
|-- masks
| |-- MCF7
| | |-- ... (assay subfolders)
| | | |-- assay_name_frame_001_mask.png
| | | |-- assay_name_frame_001_mask_C01.png
| | | -- ...
| |-- MCF10A\
| | |-- ... (assay subfolders) \
| | | |-- assay_name_frame_001_mask.png
| | | |-- assay_name_frame_001_mask_C02.png
| | | -- ...
```

## CAMAD Directory Structure
```
..\data\cmad\
|-- images\
|   |-- exp1\
|   |   |-- images\
|   |   `-- masks\
|   |-- exp2\ 
|   |   |-- images\
|   |   `-- masks\
|   `-- ... (exp3 through exp16)
|-- rois\
|   |-- exp1_roi.zip
|   |-- exp2_roi.zip
|   `-- ... (exp3 through exp16)
`-- videos\
    |-- exp1_glassmatrigel231liveimaging5aug.20fps.avi
    |-- exp2_[additional details].avi
    `-- ... (exp3 through exp16)
```

## Scripts
The `src` directory contains the scripts for visualizing the datasets and processing the images. The scripts are as follows:
- camad_extract_frames_from_video.py - Extract frames from a videos in the CAMAD given a start time, end time and the interval of frame extraction.
- roi_parser.py - Converts ImageJ ROI files from either datasets to a binary masks.
- image_overlay.py - Visualization of images and masks overlayed.
- visualization_app.py - streamlit app for visualizing images and masks.

## Usage
Clone the repository using the following command:
```bash
git clone https://github.com/leonardo-iheme/whad_camad_datasets.git
```
Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
Download the datasets from XXXX and extract the contents to the `data` directory.  
To visualize the datasets, run the streamlit app using the following command:
```bash
streamlit run code/visualization_app.py --server.enableXsrfProtection false
```
if the app does not open automatically, open a browser and navigate to `http://localhost:8501/`.

You can also run the streamlit app on the streamlit cloud using the following link: [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://whad-camad-datasets-visualization.streamlit.app/)

### Citation
If you use the datasets in your research, please cite as follows:

Iheme, L. O., Onal, S., ERDEM, Y. S., Ucar, M., Yalcin-Ozuysal, O., Pesen-Okvur, D., Behcet U., T., & Unay, D. (2024). Wound Healing Assay Dataset (WHAD) and Cell Adhesion and Motility Assay Dataset (CAMAD) (1.0.0-alpha) [Data set]. IEEE. https://doi.org/10.5281/zenodo.12806149

if you use the code in this repository, please cite the following repository:
[![DOI](https://zenodo.org/badge/833117199.svg)](https://zenodo.org/doi/10.5281/zenodo.12805890)
