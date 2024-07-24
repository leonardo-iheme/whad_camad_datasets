import streamlit as st
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
import matplotlib.colors as mcolors
from io import BytesIO


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def overlay_mask(image_path: str,
                 mask_path: str = None,
                 transparency: float = 0.2,
                 boundary: bool = False,
                 overlay_color: str = 'red',
                 save_path: str = None):
    """
    Overlay a mask on an image.
    """
    # Convert hex to RGB if necessary
    if overlay_color.startswith('#'):
        overlay_rgb = np.array(hex_to_rgb(overlay_color))
    else:
        if overlay_color not in mcolors.CSS4_COLORS:
            raise ValueError(
                f"Color '{overlay_color}' is not supported. Choose from: {', '.join(mcolors.CSS4_COLORS.keys())}")
        overlay_rgb = np.array(mcolors.to_rgb(overlay_color)) * 255

    image = Image.open(image_path)
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).resize(image.size)
        image_np = np.array(image)
        mask_np = np.array(mask)
        if len(mask_np.shape) == 2:
            mask_np = np.stack((mask_np,) * 3, axis=-1)
        if image_np.ndim == 2:
            image_np = np.stack((image_np,) * 3, axis=-1)
        mask_binary = mask_np[:, :, 0] > 0
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(mask_np[:, :, 0], cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        if boundary:
            overlay_with_boundary = image_np.copy()
            contours = measure.find_contours(mask_np[:, :, 0], 0.5)
            for contour in contours:
                contour = np.round(contour).astype(int)
                overlay_with_boundary[contour[:, 0], contour[:, 1]] = overlay_rgb
            axes[2].imshow(overlay_with_boundary.astype(np.uint8))
            axes[2].set_title('Original Image with Cell Boundaries')
        else:
            overlay = image_np.copy()
            overlay[mask_binary] = (1 - transparency) * overlay[mask_binary] + transparency * overlay_rgb
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title('Original Image with Mask Overlay')
        axes[2].axis('off')
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.array(image), cmap='gray')
        ax.set_title('Original Image')
        ax.axis('off')
    if save_path:
        plt.savefig(save_path)
    st.pyplot(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def extract_exp_number(s):
    """
    Extracts the 'exp' followed by an integer from a string.

    Parameters:
    - s (str): The string to search in.

    Returns:
    - list: A list of integers that were found following 'exp'.
    """
    # Use regular expression to find all occurrences of 'exp' followed by digits
    matches = re.findall(r'exp(\d+)', s)
    # Convert found digits to integers and return them
    return [int(match) for match in matches]


def generate_path(dataset, filename):
    """
    Generate the path for the corresponding mask or image file based on the dataset and filename.
    """
    exp_no = extract_exp_number(filename)[0]

    if dataset == 'camad':
        if ".tif" in filename:
            # Image to mask conversion for camad
            mask_filename = filename.replace(".tif", ".png")
            mask_directory = f"../data/camad/exp{exp_no}/masks/"
            return os.path.join(mask_directory, mask_filename)
        elif ".png" in filename:
            # Mask to image conversion for camad
            image_filename = filename.replace(".png", ".tif")
            image_directory = f"../data/exp{exp_no}/images/"
            return os.path.join(image_directory, image_filename)
    elif dataset == 'whad':
        if "_mask.png" in filename:
            # Mask to image conversion for whad
            image_filename = filename.replace("_mask.png", ".tif")
            image_directory = f"../data/exp{exp_no}/images/"
            return os.path.join(image_directory, image_filename)
        elif ".tif" in filename:
            # Image to mask conversion for whad
            mask_filename = filename.replace(".tif", "_mask.png")
            mask_directory = f"../data/exp{exp_no}/masks/"
            return os.path.join(mask_directory, mask_filename)
    return "Unknown pattern"


st.title("Image and Mask Overlay Visualization")

st.sidebar.header("Select Dataset")
dataset = st.sidebar.selectbox("Choose the dataset:", ["camad", "whad"])

st.sidebar.header("Upload Files")
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "tif"])
uploaded_mask = st.sidebar.file_uploader("Upload a Mask", type=["png", "jpg", "jpeg"])


def suggest_pair(uploaded_file, dataset):
    return generate_path(dataset, uploaded_file.name)


suggested_image = None
suggested_mask = None

if uploaded_image:
    suggested_mask = suggest_pair(uploaded_image, dataset)
if uploaded_mask:
    suggested_image = suggest_pair(uploaded_mask, dataset)

if suggested_image:
    st.sidebar.info(f"Suggested image: {suggested_image}")
if suggested_mask:
    st.sidebar.info(f"Suggested mask: {suggested_mask}")

if uploaded_image and uploaded_mask:
    image_path = os.path.join("temp", uploaded_image.name)
    mask_path = os.path.join("temp", uploaded_mask.name)

    if not os.path.exists("temp"):
        os.makedirs("temp")

    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    with open(mask_path, "wb") as f:
        f.write(uploaded_mask.getbuffer())

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_path, caption='Uploaded Image', use_column_width=True)
    with col2:
        st.image(mask_path, caption='Uploaded Mask', use_column_width=True)

    st.sidebar.header("Overlay Settings")
    transparency = st.sidebar.slider("Select Transparency", 0.0, 1.0, 0.2)
    boundary = st.sidebar.checkbox("Display Cell Boundaries", value=False)
    overlay_color = st.sidebar.color_picker("Select Overlay Color", "#FF0000")

    overlay_mask(image_path, mask_path, transparency, boundary, overlay_color)

    st.sidebar.header("Download")
    file_name = st.sidebar.text_input("Enter the file name for the overlay image (e.g., overlay.png):", "overlay.png")

    if st.sidebar.button("Download Overlay Image"):
        overlay_image_data = overlay_mask(image_path, mask_path, transparency, boundary, overlay_color)
        st.sidebar.download_button(label="Download Overlay Image",
                                   data=overlay_image_data,
                                   file_name=file_name,
                                   mime="image/png")
else:
    st.write("Please upload both an image and a mask to proceed.")
