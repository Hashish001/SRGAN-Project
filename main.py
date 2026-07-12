import os
import torch
import warnings
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse

warnings.filterwarnings('ignore')

from config import config
from model import Generator
from routes.super_resolve import router as super_resolve_router
from routes.compare import router as compare_router
from routes.admin import router as admin_router

# Initialize FastAPI app
app = FastAPI(title="SRGAN Super Resolution", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/original", exist_ok=True)
os.makedirs("uploads/results", exist_ok=True)
os.makedirs("uploads/logs", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the generator model
generator = Generator().to(device)

def load_model():
    """Load the trained SRGAN model"""
    if not os.path.exists(config.MODEL_PATH):
        print(f"Warning: Model file '{config.MODEL_PATH}' not found!")
        return False
    
    try:
        checkpoint = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
            elif 'state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['state_dict'])
            else:
                generator.load_state_dict(checkpoint)
        else:
            generator.load_state_dict(checkpoint)
        
        generator.eval()
        print(f"Model loaded successfully from {config.MODEL_PATH}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load model at startup
model_loaded = load_model()

# Include routers
app.include_router(super_resolve_router)
app.include_router(compare_router)
app.include_router(admin_router)

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

@app.get("/model-status")
async def model_status():
    """Get model status information"""
    return {
        "model_loaded": model_loaded,
        "device": str(device),
        "model_path": config.MODEL_PATH,
        "model_exists": os.path.exists(config.MODEL_PATH),
        "current_directory": os.getcwd(),
        "total_original_images": len(os.listdir("uploads/original")) if os.path.exists("uploads/original") else 0,
        "total_result_images": len(os.listdir("uploads/results")) if os.path.exists("uploads/results") else 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_loaded, "device": str(device)}

# Make generator and device accessible to routes
app.state.generator = generator
app.state.device = device
app.state.model_loaded = model_loaded