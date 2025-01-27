from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from PIL import Image
from django.shortcuts import render
from matplotlib.path import Path
from jpeg2000converter import settings
from utils.prepare_image import load_and_prepare_image
from utils.dwt import apply_dwt
from utils.quantization import apply_quantization
from utils.roi import apply_roi_with_highlight
from utils.save import save_image_as_jp2
from utils.check_functions import get_image_size
from utils.check_functions import print_data_size
from utils.ebcot import apply_ebcot
from utils.arithmetic_encoding import apply_arithmetic_coding
from utils.convert import convert_jp2_to_png
import os
#import logging
from utils.decode import decode_images
import uuid
from mimetypes import guess_type
from django.core.files.storage import FileSystemStorage
from django_ratelimit.decorators import ratelimit
import cv2
import numpy as np
from django.http import FileResponse
from django.shortcuts import render

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # Maks 5 MB

def home(request):
    return render(request, 'converter/home.html')

def apply_dwt_cv_rgb(image, wavelet_level):
    """
    OpenCV kullanarak RGB görüntüye DWT uygula.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:  # Control of RGB
        raise ValueError("The input image must be an RGB image.")

    transformed_channels = []
    for channel in cv2.split(image):  # Seperate RGB channels
        channel = np.float32(channel)  # DCT için float32 veri tipine dönüştür
        transformed = channel.copy()
        for _ in range(wavelet_level):
            transformed = cv2.dct(transformed)  # Apply DCT
        transformed_channels.append(transformed)

    return cv2.merge(transformed_channels)  # Kanalları birleştirerek RGB olarak geri döndür


def inverse_dwt_cv_rgb(dwt_image, wavelet_level):
    """
    OpenCV kullanarak RGB görüntüde ters DWT uygula.
    """
    if len(dwt_image.shape) != 3 or dwt_image.shape[2] != 3:  # RGB kontrolü
        raise ValueError("The input image must be an RGB image.")

    reconstructed_channels = []
    for channel in cv2.split(dwt_image):  # RGB kanalları ayır
        channel = channel.copy()
        for _ in range(wavelet_level):
            channel = cv2.idct(channel)  # Ters DCT uygula
        reconstructed_channels.append(channel)

    return cv2.merge(reconstructed_channels)  # Kanalları birleştirerek RGB olarak geri döndür

@ratelimit(key='ip', rate='7/m', block=True)
def encode(request):
    if request.method == 'POST':
        uploaded_image = request.FILES['image']
        if not uploaded_image.name.lower().endswith(('.jpeg', '.jpg')):
            return render(request, 'converter/encode.html', {
                'error': "Only JPEG format is allowed. Please upload a valid JPEG image.",
            })

        try:
            # Dosyayı medya klasörüne kaydet
            fs = FileSystemStorage()
            filename = fs.save(uploaded_image.name, uploaded_image)
            uploaded_image_path = fs.path(filename)

            # Kullanıcıdan gelen parametreleri al
            wavelet_level = int(request.POST.get('wavelet_level', 2))  # DWT seviyeleri
            threshold = float(request.POST.get('threshold', 0.01))  # Threshold değeri (0.0 - 5.0)
            compression_ratio = int(request.POST.get('compression_ratio', 30))  # Sıkıştırma oranı

            # Compression ratio kontrolü
            if compression_ratio < 10 or compression_ratio > 100:
                return render(request, 'converter/encode.html', {
                    'error': "Compression ratio must be between 20 and 100.",
                })

            # Threshold kontrolü
            if threshold < 0.0 or threshold > 5.0:
                return render(request, 'converter/encode.html', {
                    'error': "Threshold must be between 0.0 and 5.0.",
                })

            # ROI (Region of Interest) kontrolü
            try:
                roi_coords = request.POST.get('roi', None)
                if roi_coords and roi_coords.lower() != "none":
                    x, y, width, height = map(int, roi_coords.split(','))
                    roi_coords = (x, y, width, height)
                else:
                    roi_coords = None
            except ValueError as e:
                return render(request, 'converter/encode.html', {
                    'error': f"Invalid ROI format. Please enter coordinates as integers in 'x,y,width,height'. Error: {str(e)}"
                })

            # Orijinal dosya boyutunu al
            original_size = get_image_size(uploaded_image_path)

            # Görüntüyü yükle
            image = cv2.imread(uploaded_image_path, cv2.IMREAD_COLOR)  # RGB olarak yükle
            if image is None:
                raise ValueError("Image loading failed.")

            # ROI uygula: ROI bölgesi kalitesini koru, diğer bölgeleri sıkıştır
            if roi_coords:
                x, y, width, height = roi_coords
                image = apply_roi_preservation(image, roi_coords)

            # DWT işlemini uygula
            transformed_image = apply_dwt_cv_rgb(image, wavelet_level)

            # Threshold uygula
            threshold_value = threshold * 255 / 5.0  # Normalize threshold (0-255 aralığına)
            transformed_image = np.where(transformed_image > threshold_value, transformed_image, 0)

            # Ters DWT işlemini uygula
            reconstructed_image = inverse_dwt_cv_rgb(transformed_image, wavelet_level)

            # Görüntüyü sıkıştır ve kaydet (JPEG2000)
            compressed_filename = f"compressed_{uploaded_image.name}.jp2"
            compressed_path = fs.path(compressed_filename)
            compression_param = int(100 / compression_ratio)
            cv2.imwrite(
                compressed_path,
                np.uint8(reconstructed_image),
                [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_param]
            )

            # PNG önizleme dosyasını oluştur
            png_filename = f"compressed_preview_{uploaded_image.name}.png"
            png_path = fs.path(png_filename)
            cv2.imwrite(png_path, np.uint8(reconstructed_image), [cv2.IMWRITE_PNG_COMPRESSION, 9])  # PNG sıkıştırması

            # Sıkıştırılmış dosya boyutunu al
            compressed_size = get_image_size(compressed_path)

            # PNG dosyasının URL'sini oluştur
            png_url = fs.url(png_filename)

            return render(request, 'converter/encode.html', {
                'result_image': png_url,  # PNG görüntüsü tarayıcıda gösterilecek
                'original_size': f"{original_size:.2f} KB",
                'compressed_size': f"{compressed_size:.2f} KB",
                'wavelet_level': wavelet_level,
                'threshold': threshold,
                'compression_ratio': compression_ratio,
                'roi': roi_coords if roi_coords else "None",
                'compressed_image_url': fs.url(compressed_filename),  # JP2 dosyası indirilecek
            })

        except Exception as e:
            return render(request, 'converter/encode.html', {
                'error': f"An error occurred: {str(e)}",
            })

    return render(request, 'converter/encode.html', {
        'wavelet_level': 2,
        'threshold': 0.01,
        'compression_ratio': 30,
    })

def decode(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        output_format = request.POST.get('output_format', 'jpeg')  # Varsayılan olarak 'jpeg' al
        
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_image_path = fs.path(filename)
        decoded_directory = settings.MEDIA_ROOT

        # Decode işlemi
        decoded_image_path, original_size, decodedimage_size, error = decode_images(
            uploaded_image_path, decoded_directory, output_format=output_format
        )
        
        if decoded_image_path:
            decoded_image_url = fs.url(os.path.basename(decoded_image_path))
            
            # Yüzde değişim hesapla
            if original_size > 0:  # Bölme hatasını önlemek için
                  percentage_change = ((original_size - decodedimage_size) / original_size) * 100
            else:
                percentage_change = 0  # Orijinal dosya boyutu 0 ise yüzde değişim sıfır
            
            return render(request, 'converter/decode.html', {
                'result_image': decoded_image_url,
                'decoded_format': output_format,  # Seçilen format
                'original_size': f"{original_size:.2f} KB",
                'decodedimage_size': f"{decodedimage_size:.2f} KB",
                'percentage_change': f"{percentage_change:.2f}%",
            })
        else:
            return render(request, 'converter/decode.html', {'error_message': error})

    return render(request, 'converter/decode.html')


def validate_uploaded_file(uploaded_file, valid_mime_types):
    """
    Validates the uploaded file's MIME type, size, and extension.
    
    Parameters:
        uploaded_file (File): The uploaded file object.
        valid_mime_types (list): List of valid MIME types.
    
    Returns:
        str: A sanitized and unique file name for saving.
    
    Raises:
        ValueError: If the file does not meet validation criteria.
    """
    # 1. Dosya boyutunu kontrol et
    if uploaded_file.size > MAX_UPLOAD_SIZE:
        raise ValueError("File size exceeds the maximum limit of 5 MB.")
    
    # 2. MIME türünü kontrol et
    mime_type, _ = guess_type(uploaded_file.name)
    if mime_type not in valid_mime_types:
        raise ValueError(f"Unsupported file type: {mime_type}. Supported types are: {', '.join(valid_mime_types)}.")
    
    # 3. Güvenli dosya adı oluştur
    unique_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    return unique_filename

def apply_roi_preservation(image, roi_coords):
    """
    ROI bölgesinin kalitesini korur, diğer bölgeleri sıkıştırmaya hazırlar.
    """
    x, y, width, height = roi_coords

    # ROI bölgesini maske oluştur
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y:y+height, x:x+width] = 255

    # ROI dışındaki bölgeleri bulanıklaştır
    blurred = cv2.GaussianBlur(image, (15, 15), 0)

    # ROI bölgesini koru ve diğer bölgeleri sıkıştırmaya hazırla
    roi_preserved = cv2.bitwise_and(image, image, mask=mask)  # ROI'yi koru
    non_roi = cv2.bitwise_and(blurred, blurred, mask=255-mask)  # Diğer bölgeleri sıkıştırmaya hazırla

    return roi_preserved + non_roi
