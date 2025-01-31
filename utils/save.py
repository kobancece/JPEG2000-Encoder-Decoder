from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
from django.http import FileResponse

def serve_jp2_image(request, filename):
    # JP2 dosyasını aç
    jp2_path = os.path.join(settings.MEDIA_ROOT, filename)
    jp2_image = glymur.Jp2k(jp2_path)

    # JP2 dosyasını raw olarak döndür
    response = FileResponse(open(jp2_path, 'rb'), content_type='image/jp2')
    response['Content-Disposition'] = f'inline; filename="{filename}"'
    return response