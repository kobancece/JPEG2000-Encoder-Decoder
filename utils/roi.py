from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
import cv2

def apply_roi_with_highlight(image, roi_coords, roi_value=1, non_roi_value=0.5, border_thickness=5, border_color=(1, 0, 0)):
    """
    Apply ROI mask to highlight a region with optional border, keeping non-ROI regions visible but less prominent.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or RGB).
        roi_coords (tuple): Coordinates of the ROI as (x_start, y_start, width, height).
        roi_value (float): Scaling factor for ROI region.
        non_roi_value (float): Scaling factor for non-ROI regions.
        border_thickness (int): Thickness of the border around ROI.
        border_color (tuple): Color of the border in RGB.

    Returns:
        numpy.ndarray: Image with highlighted ROI.
    """
    print("\nApplying ROI mask with highlight...")
    x_start, y_start, width, height = roi_coords
    x_end, y_end = x_start + width, y_start + height

    # Create a scaled version of the image
    non_roi_image = image * non_roi_value

    # Apply scaling for ROI region
    non_roi_image[y_start:y_end, x_start:x_end] = image[y_start:y_end, x_start:x_end] * roi_value

    # Add a border around ROI
    for t in range(border_thickness):
        if len(image.shape) == 3:  # RGB image
            non_roi_image[y_start - t:y_end + t, x_start - t:x_start] = border_color  # Left border
            non_roi_image[y_start - t:y_end + t, x_end:x_end + t] = border_color      # Right border
            non_roi_image[y_start - t:y_start, x_start - t:x_end + t] = border_color  # Top border
            non_roi_image[y_end:y_end + t, x_start - t:x_end + t] = border_color      # Bottom border
        else:  # Grayscale image
            non_roi_image[y_start - t:y_end + t, x_start - t:x_start] = 1  # White border for grayscale
            non_roi_image[y_start - t:y_end + t, x_end:x_end + t] = 1
            non_roi_image[y_start - t:y_start, x_start - t:x_end + t] = 1
            non_roi_image[y_end:y_end + t, x_start - t:x_end + t] = 1

    print("ROI mask with highlight applied.")
    return non_roi_image