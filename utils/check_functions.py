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


def print_data_size(data, description=""):
    """
    Print the size of the given data in kilobytes (KB) along with a description.

    Parameters:
        data: The data whose size needs to be printed.
        description (str): A brief description of the data (e.g., "After DWT").
    """
    if isinstance(data, np.ndarray):
        # NumPy array boyutunu ölç
        size_kb = data.nbytes / 1024
        print(f"{description} - Data Size: {size_kb:.2f} KB (NumPy Array)")
    elif isinstance(data, list):
        # Liste boyutunu ölç
        size_kb = sum([np.array(item).nbytes if isinstance(item, np.ndarray) else 0 for item in data]) / 1024
        print(f"{description} - Data Size: {size_kb:.2f} KB (List of NumPy Arrays)")
    elif isinstance(data, tuple):
        # Tuple içindeki boyutları ölç
        size_kb = sum([np.array(item).nbytes if isinstance(item, np.ndarray) else 0 for item in data]) / 1024
        print(f"{description} - Data Size: {size_kb:.2f} KB (Tuple of NumPy Arrays)")
    else:
        # Diğer veri türleri
        print(f"{description} - Data Size: Cannot determine size (unsupported type)")