# Wound Healing Assay Dataset (WHAD) and Cell Adhesion and Motility Assay Dataset (CAMAD)

This repository contains the codes for visualizing and processing the Wound Healing Assay Dataset (WHAD) and Cell Adhesion and Motility Assay Dataset (CAMAD) datasets. The datasets are available at [Zenodo](https://zenodo.org/record/4640734#.YH9Q9pMzZQI).

The WHAD dataset contains images of MCF7 and MCF10A cells in 2D wound healing assays. The CAMAD dataset contains images and videos of MCF7 cells in 3D cell adhesion and motility assays.

The directory structure for the datasets is as follows:
## WHAD Directory Structure
```
..\data\whad
|-- images
| |-- MCF7
| | |-- ... (assay subfolders, e.g., "24h_EGF")
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
git clone
```
Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
To visualize the datasets, run the streamlit app using the following command:
```bash
streamlit run src/visualization_app.py --server.enableXsrfProtection false
```