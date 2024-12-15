from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
import cv2


def apply_dwt(image, wavelet='haar', levels=2):
    """
    Apply Discrete Wavelet Transform (DWT) to an image.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or RGB).
        wavelet (str): Wavelet type (default: 'haar').
        levels (int): Number of decomposition levels (default: 2).

    Returns:
        dict: Contains DWT coefficients for each channel.
    """
    print("\nApplying DWT...")
    is_grayscale = len(image.shape) == 2
    channels = ['Gray'] if is_grayscale else ['R', 'G', 'B']
    dwt_coeffs = {}

    for i, channel_name in enumerate(channels):
        channel = image if is_grayscale else image[:, :, i]
        # DWT işlemi, çıktı varsayılan olarak bir tuple dönecektir
        coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=levels, mode='symmetric')
        dwt_coeffs[channel_name] = coeffs

    return dwt_coeffs