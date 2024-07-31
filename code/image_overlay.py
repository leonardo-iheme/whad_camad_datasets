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
    :param image_path: Path to the input image.
    :param mask_path: Path to the mask image.
    :param transparency: Transparency of the mask overlay.
    :param boundary: Display the cell boundaries.
    :param overlay_color: Color of the overlay.
    :param save_path: Path to save the output image.
    :return: None
    """
    # Validate overlay color
    if overlay_color not in mcolors.CSS4_COLORS:
        raise ValueError(
            f"Color '{overlay_color}' is not supported. Choose from: {', '.join(mcolors.CSS4_COLORS.keys())}")

    overlay_rgb = np.array(mcolors.to_rgb(overlay_color)) * 255

    # Load the image
    image = Image.open(image_path)

    if mask_path and os.path.exists(mask_path):
        # Load and resize the mask
        mask = Image.open(mask_path).resize(image.size)

        # Convert images to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Convert grayscale or binary mask to RGB
        if len(mask_np.shape) == 2:
            mask_np = np.stack((mask_np,) * 3, axis=-1)

        # Ensure the image is in RGB format
        if image_np.ndim == 2:
            image_np = np.stack((image_np,) * 3, axis=-1)

        # Ensure the mask is binary
        mask_binary = mask_np[:, :, 0] > 0

        # Display the result
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0)

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
            # Create an overlay
            overlay = image_np.copy()
            overlay[mask_binary] = (1 - transparency) * overlay[mask_binary] + transparency * overlay_rgb
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title('Original Image with Mask Overlay')

        axes[2].axis('off')

    else:
        warnings.warn(f"Mask file '{mask_path}' not found. Displaying only the image.")
        # Display only the image
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.array(image), cmap='gray')
        ax.set_title('Original Image')
        ax.axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    exp = 1
    image_no = 2
    im_path = f"../data/camad/images/exp1/images/exp{exp}_{image_no}.tif"
    msk_path = f"../data/camad/images/exp1/masks/exp{exp}_{image_no}.png"
    # im_path = f"../data/whad/images/MCF7/inf1/exp2/MCF7-LacZ-p1-15-17.11.2016/MCF7_SEMA6D_wound_healing_Mark_and_Find_001_MCF7_LacZ_p001_t00_ch00.tif"
    # msk_path = im_path.replace("images", "masks").replace(".tif", "_mask.png")
    overlay_mask(im_path, msk_path, transparency=0.2, boundary=False, overlay_color='magenta', save_path='../results/overlay.png')
