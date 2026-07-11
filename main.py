import os
import io
import time
import uuid
import torch
import numpy as np
from PIL import Image, ImageEnhance
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from torchvision.transforms import ToTensor, ToPILImage
import base64
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from model import Generator

# Initialize FastAPI app
app = FastAPI(title="SRGAN Super Resolution", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Create directories for storing images
os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/original", exist_ok=True)      # صور المستخدمين الأصلية
os.makedirs("uploads/results", exist_ok=True)       # نتائج SRGAN
os.makedirs("uploads/logs", exist_ok=True)          # سجلات
os.makedirs("static/results", exist_ok=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the generator model
generator = Generator().to(device)
model_path = "SRGAN Generator Model.pth"

def load_model():
    """Load the trained SRGAN model"""
    global generator
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file '{model_path}' not found!")
        print("Please place your model file in the project directory.")
        print(f"Current directory: {os.getcwd()}")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
                print("Loaded generator state dict from checkpoint")
            elif 'state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['state_dict'])
                print("Loaded state dict from checkpoint")
            else:
                generator.load_state_dict(checkpoint)
                print("Loaded model weights directly")
        else:
            generator.load_state_dict(checkpoint)
            print("Loaded model directly")
        
        generator.eval()
        print(f"Model loaded successfully from {model_path}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print("Using untrained model")
        return False

# Load model at startup
model_loaded = load_model()

def save_upload_log(log_data):
    """Save upload log to JSON file"""
    log_file = "uploads/logs/upload_history.json"
    
    # Load existing logs
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Add new log
    logs.append(log_data)
    
    # Save logs
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

def process_image(image, enhancement_level=1.2, sharpness_level=1.1):
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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "device": str(device),
            "model_loaded": model_loaded
        }
    )

@app.post("/super-resolve")
async def super_resolve(
    request: Request,
    file: UploadFile = File(...),
    color_enhance: float = Form(1.2),
    sharpness_enhance: float = Form(1.1)
):
    """Endpoint for super resolution with image saving"""
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            return JSONResponse({
                "success": False,
                "error": "File must be an image"
            }, status_code=400)
        
        # Get client info
        client_host = request.client.host if request.client else "unknown"
        timestamp = datetime.now()
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".png"
        unique_id = uuid.uuid4().hex[:10]
        original_filename = f"original_{timestamp.strftime('%Y%m%d_%H%M%S')}_{unique_id}{file_extension}"
        result_filename = f"result_{timestamp.strftime('%Y%m%d_%H%M%S')}_{unique_id}.png"
        
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save original image to uploads/original/
        original_save_path = os.path.join("uploads", "original", original_filename)
        image.save(original_save_path)
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Original image saved: {original_save_path}")
        
        # Process image
        result, proc_time, original_size, output_size = process_image(
            image,
            enhancement_level=color_enhance,
            sharpness_level=sharpness_enhance
        )
        
        # Save result image to uploads/results/
        result_save_path = os.path.join("uploads", "results", result_filename)
        result.save(result_save_path)
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Result image saved: {result_save_path}")
        
        # Create log entry
        log_data = {
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "client_ip": client_host,
            "original_filename": file.filename,
            "saved_original": original_save_path,
            "saved_result": result_save_path,
            "original_size": f"{original_size[0]}x{original_size[1]}",
            "output_size": f"{output_size[0]}x{output_size[1]}",
            "processing_time": f"{proc_time:.2f}s",
            "color_enhancement": color_enhance,
            "sharpness_enhancement": sharpness_enhance,
            "device": str(device),
            "success": True
        }
        
        # Save log
        save_upload_log(log_data)
        
        # Convert to base64 for display
        original_b64 = image_to_base64(image)
        result_b64 = image_to_base64(result)
        
        return JSONResponse({
            "success": True,
            "original_image": original_b64,
            "result_image": result_b64,
            "processing_time": f"{proc_time:.2f}",
            "original_size": f"{original_size[0]}x{original_size[1]}",
            "output_size": f"{output_size[0]}x{output_size[1]}",
            "device": str(device),
            "model_loaded": model_loaded,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "saved_files": {
                "original": original_save_path,
                "result": result_save_path
            }
        })
    
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        
        # Log error
        error_log = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "filename": file.filename if file else "unknown",
            "error": str(e),
            "success": False
        }
        save_upload_log(error_log)
        
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/compare")
async def compare_with_bicubic(
    request: Request,
    file: UploadFile = File(...)
):
    """Compare SRGAN with bicubic upscaling"""
    try:
        # Get client info
        client_host = request.client.host if request.client else "unknown"
        timestamp = datetime.now()
        
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".png"
        unique_id = uuid.uuid4().hex[:10]
        original_filename = f"compare_original_{timestamp.strftime('%Y%m%d_%H%M%S')}_{unique_id}{file_extension}"
        result_filename = f"compare_result_{timestamp.strftime('%Y%m%d_%H%M%S')}_{unique_id}.png"
        
        # Save original image
        original_save_path = os.path.join("uploads", "original", original_filename)
        image.save(original_save_path)
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Compare original saved: {original_save_path}")
        
        # Bicubic upscaling
        w, h = image.size
        bicubic = image.resize((w * 4, h * 4), Image.LANCZOS)
        
        # SRGAN processing
        srgan_result, proc_time, _, output_size = process_image(image)
        
        # Save result
        result_save_path = os.path.join("uploads", "results", result_filename)
        srgan_result.save(result_save_path)
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Compare result saved: {result_save_path}")
        
        # Log
        log_data = {
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "client_ip": client_host,
            "type": "comparison",
            "original_filename": file.filename,
            "saved_original": original_save_path,
            "saved_result": result_save_path,
            "original_size": f"{w}x{h}",
            "output_size": f"{output_size[0]}x{output_size[1]}",
            "processing_time": f"{proc_time:.2f}s",
            "device": str(device),
            "success": True
        }
        save_upload_log(log_data)
        
        # Convert to base64
        original_b64 = image_to_base64(image)
        bicubic_b64 = image_to_base64(bicubic)
        srgan_b64 = image_to_base64(srgan_result)
        
        return JSONResponse({
            "success": True,
            "original_image": original_b64,
            "bicubic_image": bicubic_b64,
            "srgan_image": srgan_b64,
            "processing_time": f"{proc_time:.2f}",
            "original_size": f"{w}x{h}",
            "output_size": f"{output_size[0]}x{output_size[1]}",
            "device": str(device),
            "saved_files": {
                "original": original_save_path,
                "result": result_save_path
            }
        })
    
    except Exception as e:
        print(f"Error in comparison: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/view-logs")
async def view_logs():
    """Admin endpoint to view upload history"""
    log_file = "uploads/logs/upload_history.json"
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        return JSONResponse({
            "success": True,
            "total_uploads": len(logs),
            "logs": logs
        })
    else:
        return JSONResponse({
            "success": True,
            "total_uploads": 0,
            "logs": []
        })

@app.get("/view-images")
async def view_images(page: int = 1, per_page: int = 20):
    """Admin endpoint to view saved images"""
    original_dir = "uploads/original"
    results_dir = "uploads/results"
    
    images = []
    
    # Get original images
    if os.path.exists(original_dir):
        originals = sorted(os.listdir(original_dir), reverse=True)
        for img in originals:
            if img.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                images.append({
                    "type": "original",
                    "filename": img,
                    "path": os.path.join(original_dir, img),
                    "size": os.path.getsize(os.path.join(original_dir, img))
                })
    
    # Get result images
    if os.path.exists(results_dir):
        results = sorted(os.listdir(results_dir), reverse=True)
        for img in results:
            if img.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                images.append({
                    "type": "result",
                    "filename": img,
                    "path": os.path.join(results_dir, img),
                    "size": os.path.getsize(os.path.join(results_dir, img))
                })
    
    # Pagination
    total = len(images)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_images = images[start:end]
    
    return JSONResponse({
        "success": True,
        "total_images": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "images": paginated_images
    })

@app.get("/model-status")
async def model_status():
    """Get model status information"""
    return JSONResponse({
        "model_loaded": model_loaded,
        "device": str(device),
        "model_path": model_path,
        "model_exists": os.path.exists(model_path),
        "current_directory": os.getcwd(),
        "total_original_images": len(os.listdir("uploads/original")) if os.path.exists("uploads/original") else 0,
        "total_result_images": len(os.listdir("uploads/results")) if os.path.exists("uploads/results") else 0
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_loaded, "device": str(device)}

