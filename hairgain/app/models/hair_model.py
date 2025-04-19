import os
import sys
import numpy as np
from PIL import Image
import torch
from pathlib import Path

# Remove CUDA_VISIBLE_DEVICES if set to -1
if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
    del os.environ['CUDA_VISIBLE_DEVICES']

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Total GPUs available: {torch.cuda.device_count()}")
else:
    print("CUDA is not available, using CPU mode")

# Add the HairFastGAN repo to the Python path
# This assumes the HairFastGAN repository is cloned at the root level
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
hairfast_dir = os.path.join(base_dir, 'HairFastGAN')
sys.path.append(hairfast_dir)

# Import HairFast components if available
try:
    from hair_swap import HairFast, get_parser
    print(f"Successfully imported HairFastGAN from {hairfast_dir}")
except ImportError as e:
    print(f"Warning: HairFastGAN module not found. Make sure to clone the repository. Error: {e}")
    

class HairModel:
    """Interface for the HairFastGAN model"""
    
    def __init__(self):
        """Initialize the HairFast model"""
        try:
            # Check if CUDA is configured correctly
            if torch.cuda.is_available():
                print("Initializing HairFast model with CUDA support")
                device = torch.device('cuda')
                # Print all available GPUs
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print("Initializing HairFast model with CPU support")
                device = torch.device('cpu')
                
            model_args = get_parser()
            args = model_args.parse_args([])
            
            # Set device in arguments to use available GPU
            # We'll modify the args to ensure it uses the device we determined
            if hasattr(args, 'device'):
                args.device = device
            if hasattr(args, 'gpu_ids') and torch.cuda.is_available():
                # Use all available GPUs
                args.gpu_ids = list(range(torch.cuda.device_count()))
            
            self.hair_fast = HairFast(args)
            self.initialized = True
            print("HairFast model initialized successfully")
        except Exception as e:
            print(f"Error initializing HairFast model: {str(e)}")
            self.initialized = False
    
    def is_initialized(self):
        """Check if the model is initialized"""
        return self.initialized
    
    def process_images(self, face_img, shape_img, color_img, align=True, blend_threshold=0.5):
        """
        Process images with HairFastGAN model
        
        Args:
            face_img (PIL.Image): Face image
            shape_img (PIL.Image): Shape image
            color_img (PIL.Image): Color image
            align (bool): Whether to align images
            blend_threshold (float): Threshold for blending (0.0 to 1.0)
            
        Returns:
            tuple: (result_image, face_align, shape_align, color_align)
        """
        if not self.initialized:
            raise ValueError("Model not initialized")
        
        # Convert all images to RGB if needed
        face_img = self._ensure_rgb(face_img)
        shape_img = self._ensure_rgb(shape_img)
        color_img = self._ensure_rgb(color_img)
        
        # Process images with HairFast
        try:
            result = self.hair_fast.swap(face_img, shape_img, color_img, align=align, blend_threshold=blend_threshold)
            
            # If align is True, HairFast returns (result, face_align, shape_align, color_align)
            # Otherwise it returns just the result
            if align:
                result_image, face_align, shape_align, color_align = result
                return result_image, face_align, shape_align, color_align
            else:
                return result, face_img, shape_img, color_img
                
        except Exception as e:
            raise ValueError(f"Error processing images: {str(e)}")
    
    def _ensure_rgb(self, img):
        """Make sure image is in RGB format"""
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img
    
    @staticmethod
    def tensor_to_pil(tensor):
        """Convert a tensor to PIL Image"""
        if isinstance(tensor, torch.Tensor):
            # Handle different tensor formats
            if len(tensor.shape) == 4 and tensor.shape[0] == 1:  # NCHW format with batch size 1
                tensor = tensor.squeeze(0)
            
            # Convert to PIL Image
            if tensor.shape[0] == 3:  # CHW format
                from torchvision.transforms.functional import to_pil_image
                return to_pil_image(tensor)
            else:
                raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
        else:
            return tensor  # Already a PIL Image

# Create a singleton instance
print("Initializing HairModel singleton")
hair_model = HairModel() 