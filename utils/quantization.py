from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
import cv2


def apply_quantization(dwt_coeffs, original_image_shape, threshold_ratio=0.1, quantization_factors=None, wavelet='haar'):
    print("\nApplying Quantization...")
    quantized_coeffs = {}
    reconstructed_channels = []

    if quantization_factors is None:
        quantization_factors = {
            'LL': 50,
            'LH': 25,
            'HL': 25,
            'HH': 50
        }

    for channel_name, coeffs in dwt_coeffs.items():
        quantized_coeffs[channel_name] = []
        for j, coeff in enumerate(coeffs):
            if j == 0:  # LL sub-band
                quantized_coeffs[channel_name].append(coeff)  # No quantization for LL band
            else:
                sub_bands = ['LH', 'HL', 'HH']
                quantized_bands = []
                for k, sub_coeff in enumerate(coeff):
                    threshold = np.percentile(np.abs(sub_coeff), 90)
                    coeff_thresholded = pywt.threshold(sub_coeff, threshold, mode='soft')

                    quantization_factor = quantization_factors.get(sub_bands[k], 10)
                    quantized_coeff = np.round(coeff_thresholded / quantization_factor) * quantization_factor
                    quantized_bands.append(quantized_coeff)

                quantized_coeffs[channel_name].append(tuple(quantized_bands))

        reconstructed_channel = pywt.waverec2(quantized_coeffs[channel_name], wavelet)
        reconstructed_channel = np.clip(reconstructed_channel, 0, 1)
        reconstructed_channel = (reconstructed_channel * 255).astype(np.uint8)
        reconstructed_channel = reconstructed_channel[:original_image_shape[0], :original_image_shape[1]]
        downsample_factor = 3  # Use a factor of 2 or 3 for higher compression
        reconstructed_channel = reconstructed_channel[::downsample_factor, ::downsample_factor]
        reconstructed_channel = cv2.resize(reconstructed_channel, (original_image_shape[1], original_image_shape[0]))
        reconstructed_channels.append(reconstructed_channel)

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