import os
import uuid
from PIL import Image
import io
from werkzeug.utils import secure_filename

def save_uploaded_file(file, upload_folder='uploads'):
    """
    Save an uploaded file to the uploads directory
    
    Args:
        file: The uploaded file object
        upload_folder: Directory to save uploads
        
    Returns:
        str: Path to the saved file
    """
    # Ensure upload folder exists
    os.makedirs(upload_folder, exist_ok=True)
    
    # Generate a unique filename
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(upload_folder, unique_filename)
    
    # Save the file
    file.save(filepath)
    return filepath

def is_valid_image(file):
    """
    Check if the file is a valid image
    
    Args:
        file: The uploaded file object
        
    Returns:
        bool: True if the file is a valid image, False otherwise
    """
    try:
        img = Image.open(file.stream)
        img.verify()  # Verify it's an image
        file.stream.seek(0)  # Reset file pointer
        return True
    except Exception:
        return False

def open_image_from_upload(file):
    """
    Open an uploaded file as a PIL Image
    
    Args:
        file: The uploaded file object
        
    Returns:
        PIL.Image: The opened image
    """
    try:
        img = Image.open(file.stream)
        img.load()  # Load the image data
        file.stream.seek(0)  # Reset file pointer
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        return img
    except Exception as e:
        raise ValueError(f"Error opening image: {str(e)}")

def save_pil_image(img, filename, output_folder='results'):
    """
    Save a PIL Image to the output folder
    
    Args:
        img: PIL Image or tensor
        filename: Base filename
        output_folder: Directory to save results
        
    Returns:
        str: Path to the saved file
    """
    import torch
    from torchvision.utils import save_image
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate a unique filename
    base_name = os.path.splitext(filename)[0]
    unique_filename = f"{base_name}_{uuid.uuid4()}.png"
    output_path = os.path.join(output_folder, unique_filename)
    
    # Save the image
    if isinstance(img, torch.Tensor):
        save_image(img, output_path)
    else:
        img.save(output_path)
        
    return output_path 