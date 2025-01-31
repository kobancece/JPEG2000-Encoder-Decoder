import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import cv2

def apply_roi_preservation(image, roi_coords):
    x, y, width, height = roi_coords

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y:y+height, x:x+width] = 255

    blurred = cv2.GaussianBlur(image, (15, 15), 0)

    roi_preserved = cv2.bitwise_and(image, image, mask=mask)
    non_roi = cv2.bitwise_and(blurred, blurred, mask=255-mask)

    return roi_preserved + non_roi