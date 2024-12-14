from PIL import Image
import numpy as np
import glymur

def convert_jp2_to_png(jp2_path, png_path):
    """
    Convert a JPEG2000 (JP2) image to PNG format.
    
    Parameters:
        jp2_path (str): Path to the JP2 file.
        png_path (str): Path to save the converted PNG file.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # JP2 dosyasını yükle
        jp2_image = glymur.Jp2k(jp2_path)[:]
        # Normalize ederek uint8 formata çevir
        jp2_image = (jp2_image / jp2_image.max() * 255).astype(np.uint8)
        # PNG olarak kaydet
        Image.fromarray(jp2_image).save(png_path)
        print(f"JP2 successfully converted to PNG: {png_path}")
        return True
    except Exception as e:
        print(f"Error converting JP2 to PNG: {e}")
        return False
