from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
import cv2


def save_image_as_jp2(image, output_path, compression_ratios=None, tilesize=None):
    """
    Save an image in the JPEG2000 format (.jp2).

    Parameters:
        image (numpy.ndarray): Input image array (grayscale or RGB).
        output_path (str): Path where the JP2 file will be saved.
        compression_ratios (list): List of compression ratios for JPEG2000 (default: None).
        tilesize (tuple): Tile size for JPEG2000 encoding (default: None).
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Ensure 8-bit format

    if compression_ratios is None:
        compression_ratios = [10, 20, 40]  # Default compression ratios

    if tilesize:
        # Include tilesize if provided
        jp2_file = glymur.Jp2k(output_path, data=image, cratios=compression_ratios, tilesize=tilesize)
    else:
        # Default without tilesize
        jp2_file = glymur.Jp2k(output_path, data=image, cratios=compression_ratios)

    print(f"Image successfully saved with compression ratios {compression_ratios} and tilesize {tilesize}: {output_path}")


from django.http import FileResponse
import glymur

def serve_jp2_image(request, filename):
    # JP2 dosyasını aç
    jp2_path = os.path.join(settings.MEDIA_ROOT, filename)
    jp2_image = glymur.Jp2k(jp2_path)

    # JP2 dosyasını raw olarak döndür
    response = FileResponse(open(jp2_path, 'rb'), content_type='image/jp2')
    response['Content-Disposition'] = f'inline; filename="{filename}"'
    return response