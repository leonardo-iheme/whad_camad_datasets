import streamlit as st
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
import matplotlib.colors as mcolors
from io import BytesIO
from typing import Optional, List, Tuple, Union


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    """
    Convert a hex color string to an RGB tuple.

    Args:
        value (str): Hex color string.

    Returns:
        Tuple[int, int, int]: RGB tuple.
    """
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def overlay_mask(image_path: str,
                 mask_path: Optional[str] = None,
                 transparency: float = 0.2,
                 boundary: bool = False,
                 overlay_color: str = 'red',
                 save_path: Optional[str] = None) -> bytes:
    """
    Overlay a mask on an image.

    Args:
        image_path (str): Path to the input image.
        mask_path (Optional[str]): Path to the mask image.
        transparency (float): Transparency of the mask overlay.
        boundary (bool): Display the cell boundaries.
        overlay_color (str): Color of the overlay.
        save_path (Optional[str]): Path to save the output image.

    Returns:
        bytes: The overlay image in bytes.
    """
    def validate_overlay_color(overlay_color: str) -> np.ndarray:
        """Validate and convert the overlay color to RGB."""
        if overlay_color.startswith('#'):
            return np.array(hex_to_rgb(overlay_color))
        if overlay_color not in mcolors.CSS4_COLORS:
            raise ValueError(
                f"Color '{overlay_color}' is not supported. Choose from: {', '.join(mcolors.CSS4_COLORS.keys())}")
        return np.array(mcolors.to_rgb(overlay_color)) * 255

    def load_image(image_path: str) -> Image.Image:
        """Load the image."""
        return Image.open(image_path)

    def load_and_resize_mask(mask_path: str, image_size: tuple) -> Image.Image:
        """Load and resize the mask."""
        return Image.open(mask_path).resize(image_size)

    def convert_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert grayscale or binary image to RGB."""
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        return image

    def ensure_rgb_format(image: np.ndarray) -> np.ndarray:
        """Ensure the image is in RGB format."""
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        return image

    def create_overlay(image: np.ndarray, mask: np.ndarray, transparency: float, overlay_rgb: np.ndarray) -> np.ndarray:
        """Create an overlay of the mask on the image."""
        mask_binary = mask[:, :, 0] > 0
        overlay = image.copy()
        overlay[mask_binary] = (1 - transparency) * overlay[mask_binary] + transparency * overlay_rgb
        return overlay

    def display_results(image: np.ndarray, mask: np.ndarray, overlay: np.ndarray, boundary: bool, overlay_rgb: np.ndarray) -> None:
        """Display the original image, mask, and overlay."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0)

        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask[:, :, 0], cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        if boundary:
            overlay_with_boundary = image.copy()
            contours = measure.find_contours(mask[:, :, 0], 0.5)
            for contour in contours:
                contour = np.round(contour).astype(int)
                overlay_with_boundary[contour[:, 0], contour[:, 1]] = overlay_rgb
            axes[2].imshow(overlay_with_boundary.astype(np.uint8))
            axes[2].set_title('Original Image with Cell Boundaries')
        else:
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title('Original Image with Mask Overlay')

        axes[2].axis('off')
        plt.show()

    overlay_rgb = validate_overlay_color(overlay_color)
    image = load_image(image_path)

    if mask_path and os.path.exists(mask_path):
        mask = load_and_resize_mask(mask_path, image.size)
        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_np = convert_to_rgb(mask_np)
        image_np = ensure_rgb_format(image_np)
        overlay = create_overlay(image_np, mask_np, transparency, overlay_rgb)
        display_results(image_np, mask_np, overlay, boundary, overlay_rgb)
    else:
        warnings.warn(f"Mask file '{mask_path}' not found. Displaying only the image.")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.array(image), cmap='gray')
        ax.set_title('Original Image')
        ax.axis('off')
        plt.show()

    if save_path:
        plt.savefig(save_path)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    return buf.getvalue()


def extract_exp_number(s: str) -> List[int]:
    """
    Extract the 'exp' followed by an integer from a string.

    Args:
        s (str): The string to search in.

    Returns:
        List[int]: A list of integers that were found following 'exp'.
    """
    matches = re.findall(r'exp(\d+)', s)
    return [int(match) for match in matches]


def generate_path(dataset: str, filename: str) -> str:
    """
    Generate the path for the corresponding mask or image file based on the dataset and filename.

    Args:
        dataset (str): The dataset name ('camad' or 'whad').
        filename (str): The filename to generate the path for.

    Returns:
        str: The generated path.
    """
    exp_no = extract_exp_number(filename)[0]
    im_array = np.array(Image.open(filename))
    im_or_mask = 'image' if len(np.unique(im_array)) > 2 else 'mask'
    if dataset == 'camad':
        if im_or_mask == 'image':
            mask_filename = filename.replace(".tif", ".png")
            mask_directory = f"../data/camad/exp{exp_no}/masks/"
            return os.path.join(mask_directory, mask_filename)
        elif im_or_mask == 'mask':
            image_filename = filename.replace(".png", ".tif")
            image_directory = f"../data/exp{exp_no}/images/"
            return os.path.join(image_directory, image_filename)
    elif dataset == 'whad':
        if im_or_mask == 'mask':
            image_filename = filename.replace("_mask.png", ".tif")
            image_directory = f"../data/exp{exp_no}/images/"
            return os.path.join(image_directory, image_filename)
        elif im_or_mask == 'image':
            mask_filename = filename.replace(".tif", "_mask.png")
            mask_directory = f"../data/exp{exp_no}/masks/"
            return os.path.join(mask_directory, mask_filename)
    return "Unknown pattern"


st.title("Image and Mask Overlay Visualization for WHAD and CAMAD Datasets")

st.sidebar.header("Select Dataset")
dataset = st.sidebar.selectbox("Choose the dataset:", ["camad", "whad"])

st.sidebar.header("Upload Files")
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "tif"])
uploaded_mask = st.sidebar.file_uploader("Upload a Mask", type=["png", "jpg", "jpeg"])


def suggest_pair(uploaded_file, dataset):
    try:
        suggested_path = generate_path(dataset, uploaded_file.name)
        return suggested_path
    except Exception as e:
        return


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
