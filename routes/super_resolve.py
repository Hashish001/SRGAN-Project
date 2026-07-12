import os
import io
import time
import uuid
from datetime import datetime
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
import base64

from utils.image_processing import process_image, image_to_base64
from utils.logging_utils import save_upload_log

router = APIRouter()

@router.post("/super-resolve")
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
        
        # Get app state
        generator = request.app.state.generator
        device = request.app.state.device
        model_loaded = request.app.state.model_loaded
        
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
        
        # Save original image
        original_save_path = os.path.join("uploads", "original", original_filename)
        image.save(original_save_path)
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Original image saved: {original_save_path}")
        
        # Process image
        result, proc_time, original_size, output_size = process_image(
            image,
            generator,
            device,
            enhancement_level=color_enhance,
            sharpness_level=sharpness_enhance
        )
        
        # Save result image
        result_save_path = os.path.join("uploads", "results", result_filename)
        result.save(result_save_path)
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Result image saved: {result_save_path}")
        
        # Create log entry
        log_data = {
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "client_ip": client_host,
            "type": "super_resolve",
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
            "type": "super_resolve",
            "error": str(e),
            "success": False
        }
        save_upload_log(error_log)
        
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)