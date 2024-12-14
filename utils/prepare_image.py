from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
import cv2


def load_and_prepare_image(image_path):
    """
    Load, validate, and prepare an image for further processing while preserving its original color space.
    
    Parameters:
        image_path (str): Path to the image file to be loaded.
    
    Returns:
        numpy.ndarray: Processed and normalized image.
    """
    try:
        # 1. Check File Format
        valid_formats = ['jpeg', 'jpg', 'png', 'bmp']
        file_extension = os.path.splitext(image_path)[-1].lower().strip('.')
        if file_extension not in valid_formats:
            raise ValueError(f"Unsupported file format: {file_extension}. "
                             f"Only {valid_formats} formats are supported.")
        
        # 2. Load the Image
        image = Image.open(image_path)
        print(f"Image successfully loaded: {image_path}")
        print(f"Original format: {image.mode}, Size: {image.size}")
        
        # 3. Check and Convert Color Space (Preserve RGB or Grayscale)
        if image.mode == 'RGB':
            print("Image is in RGB format. Preserving RGB format...")
            image_array = np.array(image, dtype=np.float32)  # Keep RGB as-is
        elif image.mode in ['L', '1']:
            print("Image is in Grayscale format. Preserving Grayscale format...")
            image_array = np.array(image, dtype=np.float32)  # Grayscale remains as-is
        else:
            raise ValueError(f"Unknown color format: {image.mode}. Image must be RGB or Grayscale.")
        
        # 4. Check Resolution and Dimensions
        width, height = image.size
        if width < 100 or height < 100:
            raise ValueError(f"Image is too small: {width}x{height}. Minimum size must be 100x100.")
        print(f"Image dimensions are valid: {width}x{height}")
        
        # 5. Normalize Pixel Values
        if image_array.ndim == 3:  # RGB
            normalized_image = image_array / 255.0  # Normalize each channel to range 0-1
        else:  # Grayscale
            normalized_image = image_array / 255.0
        
        print("Pixel values have been normalized (range: 0-1).")
        
        # 6. Return the Processed Image
        return normalized_image
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except ValueError as ve:
        raise ValueError(f"Error: {ve}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the image: {e}")