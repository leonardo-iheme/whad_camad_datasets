import streamlit as st
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
import matplotlib.colors as mcolors
from io import BytesIO
import time

# Set page configuration with a meaningful title and layout
st.set_page_config(
    page_title="WHAD/CAMAD Visualization Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for user preferences
if 'transparency' not in st.session_state:
    st.session_state['transparency'] = 0.2
if 'boundary' not in st.session_state:
    st.session_state['boundary'] = False
if 'overlay_color' not in st.session_state:
    st.session_state['overlay_color'] = "#FF0000"
if 'last_dataset' not in st.session_state:
    st.session_state['last_dataset'] = "camad"


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


@st.cache_data
def overlay_mask(image_path: str,
                 mask_path: str = None,
                 transparency: float = 0.2,
                 boundary: bool = False,
                 overlay_color: str = 'red',
                 save_path: str = None):
    """
    Overlay a mask on an image.
    """
    try:
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
    except Exception as e:
        st.error(f"Error processing images: {str(e)}")
        return None


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
    return [int(match) for match in matches] if matches else []


def identify_file_type(file_path):
    """
    Identify if a file is an image or mask based on its content.

    Returns: 'image' or 'mask'
    """
    try:
        im_array = np.array(Image.open(file_path))
        # If image has very few unique values, it's likely a mask
        return 'mask' if len(np.unique(im_array)) <= 10 else 'image'
    except Exception:
        # If we can't determine, make a guess based on file extension
        if file_path.lower().endswith(('.png', '_mask.png')):
            return 'mask'
        elif file_path.lower().endswith(('.tif', '.jpg', '.jpeg')):
            return 'image'
        return 'unknown'


def generate_path(dataset, filename, file_path=None):
    """
    Generate the path for the corresponding mask or image file based on the dataset and filename.
    """
    try:
        exp_numbers = extract_exp_number(filename)
        if not exp_numbers:
            return None

        exp_no = exp_numbers[0]

        # Determine if this is an image or mask file
        file_type = identify_file_type(file_path) if file_path else None

        if file_type is None:
            # Try to guess from filename patterns
            if "_mask" in filename or filename.endswith(".png"):
                file_type = 'mask'
            elif filename.endswith(".tif"):
                file_type = 'image'
            else:
                return None

        if dataset == 'camad':
            if file_type == 'image':
                # Image to mask conversion for camad
                mask_filename = filename.replace(".tif", ".png")
                mask_directory = f"../data/camad/exp{exp_no}/masks/"
                return os.path.join(mask_directory, mask_filename)
            elif file_type == 'mask':
                # Mask to image conversion for camad
                image_filename = filename.replace(".png", ".tif")
                image_directory = f"../data/exp{exp_no}/images/"
                return os.path.join(image_directory, image_filename)
        elif dataset == 'whad':
            if file_type == 'mask':
                # Mask to image conversion for whad
                image_filename = filename.replace("_mask.png", ".tif")
                image_directory = f"../data/exp{exp_no}/images/"
                return os.path.join(image_directory, image_filename)
            elif file_type == 'image':
                # Image to mask conversion for whad
                mask_filename = filename.replace(".tif", "_mask.png")
                mask_directory = f"../data/exp{exp_no}/masks/"
                return os.path.join(mask_directory, mask_filename)
        return None
    except Exception as e:
        st.sidebar.warning(f"Could not suggest matching file: {str(e)}")
        return None


def create_temp_dir():
    """Create the temporary directory if it does not exist"""
    temp_dir = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(temp_dir):
        try:
            os.makedirs(temp_dir)
        except Exception as e:
            st.error(f"Failed to create temporary directory: {str(e)}")
    return temp_dir


def save_uploaded_file(uploaded_file):
    """Save an uploaded file to the temp directory and return the path"""
    temp_dir = create_temp_dir()
    file_path = os.path.join(temp_dir, uploaded_file.name)

    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Failed to save uploaded file: {str(e)}")
        return None


def generate_overlay(image_path, mask_path, transparency, boundary, overlay_color):
    """Generate overlay and display it with timing information"""
    if image_path and mask_path:
        with st.spinner("Generating overlay..."):
            start_time = time.time()
            overlay_result = overlay_mask(
                image_path,
                mask_path,
                transparency,
                boundary,
                overlay_color
            )
            end_time = time.time()
            if overlay_result:
                st.success(f"Overlay generated in {end_time - start_time:.2f} seconds")
                # Add a download button
                st.download_button(
                    label="Download Overlay Image",
                    data=overlay_result,
                    file_name=f"overlay_{int(time.time())}.png",
                    mime="image/png",
                )
            return overlay_result
    return None


def main():
    """Main function to run the app"""

    # App title with a divider for clean separation
    st.title("🔬 Image and Mask Overlay Visualization Tool")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.sidebar.header("🔧 Configuration")

        # Dataset selection with remembered value
        dataset = st.selectbox(
            "Choose the dataset:",
            ["camad", "whad"],
            index=0 if st.session_state['last_dataset'] == "camad" else 1
        )
        st.session_state['last_dataset'] = dataset

        st.markdown("---")
        st.header("📁 Upload Files")

        # File uploaders with clear instructions
        uploaded_image = st.file_uploader(
            "Upload an Image",
            type=["png", "jpg", "jpeg", "tif"],
            help="Upload a microscope image file (.tif, .png, .jpg)"
        )

        uploaded_mask = st.file_uploader(
            "Upload a Mask",
            type=["png", "jpg", "jpeg"],
            help="Upload a segmentation mask file (typically .png)"
        )

    # Process uploads and suggest pairs
    image_path = None
    mask_path = None
    suggested_image = None
    suggested_mask = None

    # Save uploaded files and generate suggestions
    if uploaded_image:
        image_path = save_uploaded_file(uploaded_image)
        if image_path:
            suggested_mask = generate_path(dataset, uploaded_image.name, image_path)

    if uploaded_mask:
        mask_path = save_uploaded_file(uploaded_mask)
        if mask_path:
            suggested_image = generate_path(dataset, uploaded_mask.name, mask_path)

    # Show suggestions in sidebar if available
    with st.sidebar:
        if suggested_image:
            st.info(f"📎 Suggested image: {os.path.basename(suggested_image)}")
            if st.button("Use Suggested Image"):
                # Here you would need to load the suggested image
                st.info("Image suggestion feature will load the file when implemented")

        if suggested_mask:
            st.info(f"📎 Suggested mask: {os.path.basename(suggested_mask)}")
            if st.button("Use Suggested Mask"):
                # Here you would need to load the suggested mask
                st.info("Mask suggestion feature will load the file when implemented")

    # Main display area
    if uploaded_image or uploaded_mask:
        # Display uploaded files in columns
        cols = st.columns(2)

        with cols[0]:
            if uploaded_image:
                st.subheader("📸 Uploaded Image")
                st.image(image_path, use_container_width=True)
                st.caption(f"Filename: {uploaded_image.name}")
            else:
                st.info("👈 Please upload an image file")

        with cols[1]:
            if uploaded_mask:
                st.subheader("🔍 Uploaded Mask")
                st.image(mask_path, use_container_width=True)
                st.caption(f"Filename: {uploaded_mask.name}")
            else:
                st.info("👈 Please upload a mask file")

    # Overlay settings section - only show if both files are uploaded
    if uploaded_image and uploaded_mask:
        st.markdown("---")
        st.subheader("⚙️ Overlay Settings")

        # Organize settings in columns for better space usage
        col1, col2, col3 = st.columns(3)

        with col1:
            transparency = st.slider(
                "Transparency",
                0.0, 1.0,
                st.session_state['transparency'],
                help="Adjust the transparency of the mask overlay",
                key="transparency_slider"
            )
            st.session_state['transparency'] = transparency

        with col2:
            boundary = st.checkbox(
                "Show boundary only",
                st.session_state['boundary'],
                help="Show only the boundary of the mask instead of the filled mask",
                key="boundary_checkbox"
            )
            st.session_state['boundary'] = boundary

        with col3:
            overlay_color = st.color_picker(
                "Overlay Color",
                st.session_state['overlay_color'],
                help="Choose the color for the mask overlay",
                key="overlay_color_picker"
            )
            st.session_state['overlay_color'] = overlay_color

        # Automatically generate overlay when both files are uploaded or settings change
        generate_overlay(image_path, mask_path, transparency, boundary, overlay_color)


if __name__ == "__main__":
    main()
