from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
import cv2

def generate_codestream(arithmetic_compressed_streams, image_shape, block_size):
    """
    Generate codestream for the compressed data.

    Parameters:
        arithmetic_compressed_streams (list): Streams after arithmetic coding.
        image_shape (tuple): Original image shape (height, width, channels).
        block_size (int): Size of the coding blocks.

    Returns:
        bytes: Codestream containing header and compressed data.
    """
    print("\nGenerating codestream...")
    
    # 1. Create codestream header
    height, width = image_shape[:2]
    num_blocks = len(arithmetic_compressed_streams)
    header = {
        "magic": "JPEG2000",  # Identifier
        "width": width,
        "height": height,
        "block_size": block_size,
        "num_blocks": num_blocks,
    }

    # Convert header to bytes
    header_bytes = (
        f"{header['magic']}|{header['width']}|{header['height']}|"
        f"{header['block_size']}|{header['num_blocks']}\n"
    ).encode("utf-8")

    # 2. Append compressed streams
    compressed_data = b""
    for idx, stream in enumerate(arithmetic_compressed_streams):
        # Convert each stream to bytes and append to the compressed data
        block_header = f"Block {idx}:\n".encode("utf-8")
        block_data = bytes(stream)
        compressed_data += block_header + block_data + b"\n"

    # Combine header and compressed data
    codestream = header_bytes + compressed_data

    print("Codestream generated successfully.")
    return codestream