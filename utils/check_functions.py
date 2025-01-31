from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
import cv2


def get_image_size(image_path):
    """
    Get the size of an image file in kilobytes (KB).

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        float: File size in KB.
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return None

    file_size_bytes = os.path.getsize(image_path)
    file_size_kb = file_size_bytes / 1024
    return file_size_kb