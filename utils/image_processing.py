import io
import time
import base64
import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import ToTensor, ToPILImage

def process_image(image, generator, device, enhancement_level=1.2, sharpness_level=1.1):
    """Process image through SRGAN model"""
    start_time = time.time()
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_size = image.size
    
    # Resize to multiples of 4 for optimal processing
    w, h = image.size
    new_w = (w // 4) * 4
    new_h = (h // 4) * 4
    
    if new_w == 0 or new_h == 0:
        new_w = max(4, w)
        new_h = max(4, h)
    
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Convert to tensor and normalize
    tensor = ToTensor()(image).unsqueeze(0).to(device)
    tensor = tensor * 2 - 1  # Normalize to [-1, 1]
    
    # Run inference
    with torch.no_grad():
        output = generator(tensor)
    
    # Convert back to image
    output = (output + 1) / 2
    output = torch.clamp(output, 0, 1)
    result = ToPILImage()(output.squeeze().cpu())
    
    output_size = result.size
    
    # Apply enhancements
    if enhancement_level != 1.0:
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(enhancement_level)
    
    if sharpness_level != 1.0:
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(sharpness_level)
    
    # Apply contrast enhancement
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(1.05)
    
    processing_time = time.time() - start_time
    
    return result, processing_time, original_size, output_size

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str