from PIL import Image
import numpy as np
import glymur
from mimetypes import guess_type
import uuid

MAX_UPLOAD_SIZE = 5 * 1024 * 1024

def validate_uploaded_file(uploaded_file, valid_mime_types):
    if uploaded_file.size > MAX_UPLOAD_SIZE:
        raise ValueError("File size exceeds the maximum limit of 5 MB.")
    mime_type, _ = guess_type(uploaded_file.name)
    if mime_type not in valid_mime_types:
        raise ValueError(f"Unsupported file type: {mime_type}. Supported types are: {', '.join(valid_mime_types)}.")
    unique_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    return unique_filename
