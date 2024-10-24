import os
import warnings
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import matplotlib.colors as mcolors
from typing import Optional


def overlay_mask(image_path: str,
                 mask_path: Optional[str] = None,
                 transparency: float = 0.2,
                 boundary: bool = False,
                 overlay_color: str = 'red',
                 save_path: Optional[str] = "output.png") -> None:
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
        None
    """
    def validate_overlay_color(overlay_color: str) -> np.ndarray:
        """Validate and convert the overlay color to RGB."""
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


if __name__ == '__main__':
    exp = 1
    image_no = 2
    im_path = f"../data/camad/images/exp1/images/exp{exp}_{image_no}.tif"
    msk_path = f"../data/camad/images/exp1/masks/exp{exp}_{image_no}.png"
    overlay_mask(im_path, msk_path, transparency=0.2, boundary=False, overlay_color='magenta', save_path='../results/overlay.png')
