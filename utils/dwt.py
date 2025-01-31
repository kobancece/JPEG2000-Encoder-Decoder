from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
import cv2


def apply_dwt_cv_rgb(image, wavelet_level):
    """
    Apply DWT to RGB image using OpenCV.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:  # Control of RGB
        raise ValueError("The input image must be an RGB image.")

    transformed_channels = []
    for channel in cv2.split(image):  # Seperate RGB channels
        channel = np.float32(channel)
        transformed = channel.copy()
        for _ in range(wavelet_level):
            transformed = cv2.dct(transformed)  # Apply DCT
        transformed_channels.append(transformed)

    return cv2.merge(transformed_channels)


def inverse_dwt_cv_rgb(dwt_image, wavelet_level):
    """
    Apply inverse DWT on RGB image using OpenCV.
    """
    if len(dwt_image.shape) != 3 or dwt_image.shape[2] != 3:  # RGB control
        raise ValueError("The input image must be an RGB image.")

    reconstructed_channels = []
    for channel in cv2.split(dwt_image):  # Seperate RGB channels
        channel = channel.copy()
        for _ in range(wavelet_level):
            channel = cv2.idct(channel)  # apply inverse DCT
        reconstructed_channels.append(channel)

    return cv2.merge(reconstructed_channels)