import os
import json
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/view-logs")
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

@router.get("/view-images")
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