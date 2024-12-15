from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
import cv2


def apply_quantization(dwt_coeffs, original_image_shape, wavelet='haar', threshold_ratio=1.5):
    """
    Apply adaptive quantization to DWT coefficients and reconstruct the image.

    Parameters:
        dwt_coeffs (dict): DWT coefficients for each channel.
        original_image_shape (tuple): Shape of the original image.
        wavelet (str): Wavelet type used for reconstruction.
        threshold_ratio (float): Scaling factor for threshold calculation.

    Returns:
        dict: Quantized DWT coefficients.
        numpy.ndarray: Reconstructed image.
    """
    quantized_coeffs = {}
    reconstructed_channels = []

    for channel_name, coeffs in dwt_coeffs.items():
        quantized_coeffs[channel_name] = []

        for i, coeff in enumerate(coeffs):
            if i == 0:  # LL band
                quantized_coeffs[channel_name].append(coeff)  # LL bandı kuantize edilmeden bırakılıyor
            else:
                sub_bands = ['LH', 'HL', 'HH']
                quantized_bands = []
                for k, sub_coeff in enumerate(coeff):  # Alt bandı tek tek işle
                    threshold = np.median(np.abs(sub_coeff)) * threshold_ratio  # threshold_ratio burada kullanılıyor
                    coeff_thresholded = pywt.threshold(sub_coeff, threshold, mode='soft')
                    quantization_factor = 10  # Sabit bir faktör
                    quantized_band = np.round(coeff_thresholded / quantization_factor) * quantization_factor
                    quantized_bands.append(quantized_band)
                quantized_coeffs[channel_name].append(tuple(quantized_bands))  # Alt bantları tuple olarak ekle

        # Kanalı yeniden yapılandır
        reconstructed_channel = pywt.waverec2(quantized_coeffs[channel_name], wavelet)
        reconstructed_channel = np.clip(reconstructed_channel, 0, 1)
        reconstructed_channel = (reconstructed_channel * 255).astype(np.uint8)
        reconstructed_channel = reconstructed_channel[:original_image_shape[0], :original_image_shape[1]]
        reconstructed_channels.append(reconstructed_channel)

    # Tüm kanalları birleştir
    if len(reconstructed_channels) == 1:
        reconstructed_image = reconstructed_channels[0]
    else:
        reconstructed_image = np.stack(reconstructed_channels, axis=2)

    return quantized_coeffs, reconstructed_image


def reconstruct_image(quantized_coeffs, print_data_size='haar'):
    """
    Reconstruct the image from quantized DWT coefficients.

    Parameters:
        quantized_coeffs (dict): Quantized DWT coefficients for each channel.
        wavelet (str): Wavelet type (default: 'haar').

    Returns:
        numpy.ndarray: Reconstructed image.
    """
    print("\nReconstructing Image...")
    is_grayscale = len(quantized_coeffs) == 1
    reconstructed_channels = []

    for channel_name, coeffs in quantized_coeffs.items():
        reconstructed_channel = pywt.waverec2(coeffs, wavelet)
        reconstructed_channel = np.clip(reconstructed_channel, 0, 1)
        reconstructed_channels.append(reconstructed_channel)

    if is_grayscale:
        reconstructed_image = reconstructed_channels[0]
    else:
        reconstructed_image = np.stack(reconstructed_channels, axis=2)

    reconstructed_image = reconstruct_image_from_quantized_coeffs(quantized_coeffs, wavelet)
    return quantized_coeffs, reconstructed_image