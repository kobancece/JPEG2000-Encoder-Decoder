from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Supported formats
SUPPORTED_FORMATS = {"png", "jpeg", "jpg", "bmp"}

def decode_images(encoded_file_path, decoded_dir, output_format="jpeg"):
    try:
        # Dosya uzantısını kontrol et
        if not encoded_file_path.lower().endswith('.jp2'):
            raise ValueError("Only JPEG2000 files (.jp2) are supported for decoding.")
        
        # Orijinal dosyanın boyutunu al
        original_size = os.path.getsize(encoded_file_path) / 1024  # KB cinsinden

        # JPEG2000 dosyasını aç
        image = Image.open(encoded_file_path)
        
        # Çıktı formatını kontrol et
        if output_format not in {"jpeg", "png", "bmp"}:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Çözülen dosyanın kaydedileceği yolu belirle
        decoded_file_path = Path(decoded_dir) / f"{Path(encoded_file_path).stem}.{output_format}"
        
        # Resmi seçilen formatta kaydet
        image.save(decoded_file_path, format=output_format.upper())
        
        # Decode edilmiş dosyanın boyutunu al
        decodedimage_size = os.path.getsize(decoded_file_path) / 1024  # KB cinsinden
        
        return decoded_file_path, original_size, decodedimage_size, None  # Başarılı olduğunda tüm bilgileri döndür
    except Exception as e:
        return None, None, None, str(e)  # Hata durumunda None ve hata mesajı döner
