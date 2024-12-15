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
import glymur
import numpy as np
#import logging
from utils.decode import decode_images

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)  
#logger.debug('Debug mesajı')
#logger.info('Info mesajı')
#logger.warning('Warning mesajı')
#logger.error('Error mesajı')


def home(request):
    return render(request, 'converter/home.html')

def encode(request):
    if request.method == 'POST':
        # Retrieve user inputs
        uploaded_image = request.FILES['image']
        
        # Check file format
        if not uploaded_image.name.lower().endswith(('.jpeg', '.jpg')):
            return render(request, 'converter/encode.html', {
                'error': "Only JPEG format is allowed. Please upload a valid JPEG image.",
            })
            
        wavelet_level = int(request.POST.get('wavelet_level', 2))
        compression_ratios = list(map(int, request.POST.get('compression_ratios', '40,30').split(',')))
        tile_size = tuple(map(int, request.POST.get('tile_size', '128,128').split(',')))
        threshold = float(request.POST.get('threshold', 0.01))
        roi = request.POST.get('roi', None)

        # Save uploaded image
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        uploaded_image_path = fs.path(filename)

        # Get original image size
        original_size = get_image_size(uploaded_image_path)

        # Prepare the image
        processed_image = load_and_prepare_image(uploaded_image_path)

        # Apply ROI if provided
        try:
            if roi and roi != "None":
                roi_coords = tuple(map(int, roi.split(',')))  # Convert to integers
                if len(roi_coords) != 4:
                    raise ValueError("ROI must have exactly 4 values (x, y, width, height).")
                processed_image = apply_roi_with_highlight(processed_image, roi_coords)
        except ValueError as e:
            return render(request, 'converter/encode.html', {
                'error': f"Invalid ROI format: {e}",
                'wavelet_level': wavelet_level,
                'compression_ratios': ','.join(map(str, compression_ratios)),
                'tile_size': ','.join(map(str, tile_size)),
                'threshold': threshold,
                'roi': roi,
            })


        # Apply DWT
        dwt_results = apply_dwt(processed_image, wavelet='haar', levels=wavelet_level)

        # Apply Quantization
        quantized_coeffs, reconstructed_image = apply_quantization(
            dwt_results, processed_image.shape, threshold_ratio=threshold, wavelet='haar'
        )

        # Apply EBCOT
        # ebcot_streams = apply_ebcot(reconstructed_image, block_size=64, target_rate=0.5)

        # Apply Arithmetic Coding
        #arithmetic_compressed_streams = apply_arithmetic_coding(ebcot_streams)

        # Save as JPEG2000
        compressed_filename = f"compressed_{uploaded_image.name}.jp2"
        compressed_path = fs.path(compressed_filename)
        save_image_as_jp2(reconstructed_image, compressed_path, compression_ratios=compression_ratios, tilesize=tile_size)

        # JP2 dosyasını PNG'ye dönüştür
        png_filename = f"compressed_preview_{uploaded_image.name}.png"
        png_path = fs.path(png_filename)
        if convert_jp2_to_png(compressed_path, png_path):
            png_url = fs.url(png_filename)
        else:
            png_url = None

        # Get compressed image size
        compressed_size = get_image_size(compressed_path)

        # Render template with all inputs and results
        return render(request, 'converter/encode.html', {
            'result_image': png_url,  # PNG görüntüsü tarayıcıda gösterilecek
            'original_size': f"{original_size:.2f} KB",
            'compressed_size': f"{compressed_size:.2f} KB",
            'wavelet_level': wavelet_level,
            'compression_ratios': ','.join(map(str, compression_ratios)),
            'tile_size': ','.join(map(str, tile_size)),
            'threshold': threshold,
            'roi': roi,
            'compressed_image_url': fs.url(compressed_filename),  # JP2 dosyası indirilecek
        })

    return render(request, 'converter/encode.html', {
        'wavelet_level': 2,
        #'compression_ratios': '10,20,40',
        'tile_size': '128,128',
        'threshold': 0.01,
        'roi': '',
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