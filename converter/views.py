from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from PIL import Image
import os
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from matplotlib.path import Path
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
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from PIL import Image
import glymur
import matplotlib.pyplot as plt
import numpy as np

def home(request):
    return render(request, 'converter/home.html')

def encode(request):
    if request.method == 'POST':
        # Retrieve user inputs
        uploaded_image = request.FILES['image']
        wavelet_level = int(request.POST.get('wavelet_level', 2))
        compression_ratios = list(map(int, request.POST.get('compression_ratios', '30,20').split(',')))
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
        if roi:
            try:
                roi_coords = tuple(map(int, roi.split(',')))
                processed_image = apply_roi_with_highlight(processed_image, roi_coords)
            except Exception as e:
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
        ebcot_streams = apply_ebcot(reconstructed_image, block_size=64, target_rate=0.5)

        # Apply Arithmetic Coding
        arithmetic_compressed_streams = apply_arithmetic_coding(ebcot_streams)

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


SUPPORTED_FORMATS = {"png", "jpeg", "jpg", "bmp"}  # Desteklenen formatlar

def decode_images(uploaded_file, decoded_dir):
    try:
        print(f"Uploaded file name: {uploaded_file.name}")

        # Check unsupported format
        if not uploaded_file.name.endswith('.jp2'):
            print("File is not a .jp2 file.")
            return None, "Invalid file format. Please upload a valid JPEG2000 (.jp2) file."

        # Create the directory for decoded images if it doesn't exist
        decoded_dir = Path(decoded_dir)
        decoded_dir.mkdir(parents=True, exist_ok=True)
        print(f"Decoded directory created or already exists: {decoded_dir}")

        # Save the uploaded file temporarily
        temp_path = decoded_dir / uploaded_file.name
        print(f"Temporary file path: {temp_path}")
        with open(temp_path, 'wb+') as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
        print(f"File saved temporarily at: {temp_path}")

        # Check if the uploaded file is in JPEG2000 format
        with Image.open(temp_path) as img:
            print(f"Image format detected: {img.format}")
            if img.format != "JPEG2000":
                print("File is not in JPEG2000 format. Deleting temporary file.")
                temp_path.unlink()  # Remove the file if it's not valid
                return None, "Invalid file format. Please upload a valid JPEG2000 file."

            # Determine the original format from filename
            decoded_extension = uploaded_file.name.split(".")[0].split("_")[-1].lower()
            print(f"Original format detected from filename: {decoded_extension}")

            # If format is unsupported, default to PNG
            if decoded_extension not in SUPPORTED_FORMATS:
                decoded_extension = "png"  # Default format
                warning_message = "Format could not be detected. Converted to PNG."
                print(warning_message)
            else:
                warning_message = None
                print(f"Detected format is supported: {decoded_extension}")

            # Decode and save the image with the detected or default format
            decoded_path = decoded_dir / f"{temp_path.stem}_decoded.{decoded_extension}"
            img.save(decoded_path, format=decoded_extension.upper())
            print(f"Decoded file saved at: {decoded_path}")
            return decoded_path, warning_message  # Return path and optional warning

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, str(e)  # Return error message if any



def decode(request):
    if request.method == 'POST' and 'image' in request.FILES:
        uploaded_file = request.FILES['image']
        print(f"Uploaded file: {uploaded_file.name}")
        
        # Specify the directory for decoded files
        decoded_directory = 'media/decoded/'
        print(f"Decoded directory: {decoded_directory}")

        # Call the decoding function
        decoded_image_path, warning_message = decode_images(uploaded_file, decoded_directory)
        print(f"Decoded image path: {decoded_image_path}")
        print(f"Warning message: {warning_message}")

        if not decoded_image_path:
            print("Decoding failed. Returning an error message to the template.")
            # Return error if decoding failed
            return render(request, 'converter/decode.html', {
                'error_message': "An error occurred while decoding the file."
            })

        # Provide the decoded image URL to the template
        fs = FileSystemStorage(location=decoded_directory)
        decoded_image_url = fs.url(os.path.basename(decoded_image_path))
        print(f"Decoded image URL: {decoded_image_url}")
    
        # Extract the decoded format from the file extension
        decoded_format = decoded_image_path.suffix[1:]  # Extract extension without the dot
        print(f"Decoded format: {decoded_format}")

        return render(request, 'converter/decode.html', {
            'result_image': decoded_image_url,
            'decoded_format': decoded_format,  # Send the format to the HTML template
            'warning_message': warning_message  # Send any warning to the HTML template
        })

    print("Request is not POST or does not contain an 'image'. Rendering the default decode page.")
    return render(request, 'converter/decode.html')


