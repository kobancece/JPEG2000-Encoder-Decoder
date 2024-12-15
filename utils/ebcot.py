from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
import cv2

def apply_ebcot(image, block_size=64, target_rate=0.5, wavelet_levels=2):
    """
    EBCOT algoritmasını tek bir fonksiyon içinde uygular.
    
    Parameters:
        image (numpy.ndarray): Giriş görüntüsü (grayscale veya RGB).
        block_size (int): Kodlama bloklarının boyutu (örneğin, 64x64).
        target_rate (float): Hedef sıkıştırma oranı (örneğin, 0.5).
        wavelet_levels (int): DWT seviyeleri (default: 2).
    
    Returns:
        list: Trankasyondan geçmiş tüm bit akışları.
    """
    def split_into_blocks(image, block_size):
        """
        Görüntüyü bloklara böler.
        """
        height, width = image.shape[:2]
        blocks = []

        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                block = image[y:y+block_size, x:x+block_size]
                blocks.append(block)
        
        return blocks

    def bit_plane_extraction(block):
        """
        Bir bloktaki bit planlarını çıkarır.
        """
        max_value = block.max()
        num_bits = int(np.ceil(np.log2(max_value + 1)))
        bit_planes = []

        for bit in range(num_bits):
            bit_plane = (block >> bit) & 1
            bit_planes.append(bit_plane)
        
        return bit_planes

    def encode_bit_planes(bit_planes):
        """
        Encode bit planes using EBCOT with context modeling.
    
        Parameters:
            bit_planes (list of numpy.ndarray): List of bit-planes to encode.
    
        Returns:
            list: Encoded bit-streams.
        """
        coded_streams = []
    
        for plane in bit_planes:
            # 1. Significance Propagation
            updated_plane = significance_propagation(plane)
    
            # 2. Magnitude Refinement (dummy işlem, iyileştirilebilir)
            magnitude_stream = magnitude_refinement(updated_plane)
    
            # 3. Cleanup Pass (dummy işlem, iyileştirilebilir)
            cleanup_stream = cleanup_pass(magnitude_stream)
    
            # Kodlanmış bit akışını ekle
            coded_streams.append(cleanup_stream)
    
        return coded_streams

    def significance_propagation(plane):
        """
        Perform significance propagation with context modeling.
    
        Parameters:
            plane (numpy.ndarray): The bit-plane to process.
    
        Returns:
            numpy.ndarray: The updated bit-plane after significance propagation.
        """
        # 1. Bağlam Modeli Matrisini Oluştur
        context_model = np.zeros_like(plane)
    
        # 2. Her Piksel İçin Komşuluk Bilgisi Hesapla
        for y in range(1, plane.shape[0] - 1):  # Kenarlardan kaçın
            for x in range(1, plane.shape[1] - 1):
                # Komşu piksellerin toplamını bağlam modeli olarak hesapla
                context_model[y, x] = np.sum(plane[y-1:y+2, x-1:x+2]) - plane[y, x]
    
        # 3. Bağlama Göre Düzeltme Uygula
        updated_plane = plane * (context_model > 0)
    
        return updated_plane

    def magnitude_refinement(stream):
        # Örnek kod: Bu geçiş için gerçek algoritma uygulanmalıdır.
        return stream

    def cleanup_pass(stream):
        # Örnek kod: Bu geçiş için gerçek algoritma uygulanmalıdır.
        return stream

    def optimal_truncation(coded_streams, target_rate):
        """
        Optimal trankasyon uygular.
        """
        truncated_streams = []
        for stream in coded_streams:
            truncated_stream = stream[:int(len(stream) * target_rate)]
            truncated_streams.append(truncated_stream)
        
        return truncated_streams

    # 1. Görüntüyü bloklara ayırma
    blocks = split_into_blocks(image, block_size)
    all_coded_streams = []

    # 2. Her blok için EBCOT işlemleri
    for block in blocks:
        # Bit planlarını çıkar
        bit_planes = bit_plane_extraction(block)
        # Kodlama geçişlerini uygula
        coded_streams = encode_bit_planes(bit_planes)
        # Optimal trankasyon uygula
        truncated_streams = optimal_truncation(coded_streams, target_rate)
        all_coded_streams.append(truncated_streams)
    return all_coded_streams