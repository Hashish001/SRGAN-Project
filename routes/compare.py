import os
import io
from datetime import datetime
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import JSONResponse
import uuid

from utils.image_processing import process_image, image_to_base64
from utils.logging_utils import save_upload_log

router = APIRouter()

@router.post("/compare")
async def compare_with_bicubic(
    request: Request,
    file: UploadFile = File(...)
):
    """Compare SRGAN with bicubic upscaling"""
    try:
        # Get app state
        generator = request.app.state.generator
        device = request.app.state.device
        
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
        srgan_result, proc_time, _, output_size = process_image(image, generator, device)
        
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