from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import requests
import tempfile
import os
import sys
import gc
import logging
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import VitalSigns class
try:
    from predict_vitals import VitalSigns
    logger.info("Successfully imported VitalSigns")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure predict_vitals.py is in the same directory")
    sys.exit(1)

import numpy as np

# -----------------------
# Pydantic models
# -----------------------
class VitalSignsRequest(BaseModel):
    video_url: str
    age: Optional[int] = 25
    gender: Optional[str] = 'M'
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 1 or v > 120:
            raise ValueError('Age must be between 1 and 120')
        return v
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v.upper() not in ['M', 'F']:
            raise ValueError('Gender must be M or F')
        return v.upper()

class VitalSignsResponse(BaseModel):
    heart_rate: Optional[float]
    systolic_bp: Optional[float]
    diastolic_bp: Optional[float]
    spo2: Optional[float]
    confidence: str
    frames_processed: Optional[int]
    error: Optional[str]
    processing_time: Optional[float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: Optional[str] = None

# -----------------------
# Global predictor
# -----------------------
vital_signs_predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vital_signs_predictor
    logger.info("Initializing Vital Signs Predictor...")
    try:
        vital_signs_predictor = VitalSigns()
        logger.info("Vital Signs Predictor initialized successfully")
        
        # Force garbage collection after initialization
        gc.collect()
        
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        vital_signs_predictor = None
    yield
    logger.info("Shutting down...")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(
    title="Vital Signs Prediction API",
    description="API for predicting vital signs from video URLs",
    version="1.0.0",
    lifespan=lifespan
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

# -----------------------
# Video Processor
# -----------------------
class VideoProcessor:
    """Helper class to handle video download and cleanup"""
    
    async def download_video(self, video_url: str) -> str:
        """Download video from URL to temporary file"""
        try:
            logger.info(f"Downloading video from: {video_url}")
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "temp_video.mp4")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(
                video_url, 
                stream=True, 
                timeout=120,  # Increased timeout
                headers=headers
            )
            response.raise_for_status()
            
            # Write video file in chunks
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(temp_file)
            logger.info(f"Video downloaded successfully. Size: {file_size / 1024 / 1024:.2f} MB")
            return temp_file
            
        except requests.RequestException as e:
            logger.error(f"Failed to download video: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")

    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary files"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                temp_dir = os.path.dirname(file_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                logger.info(f"Cleaned up temporary file: {file_path}")
                
                # Force garbage collection after cleanup
                gc.collect()
                
        except Exception as e:
            logger.warning(f"Could not clean up temp file {file_path}: {e}")

video_processor = VideoProcessor()

# -----------------------
# Exception handlers
# -----------------------
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "available_endpoints": [
                "GET /",
                "GET /health",
                "POST /predict",
                "GET /status",
                "GET /docs"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# -----------------------
# Routes
# -----------------------
@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.head("/", tags=["Health"])
async def root():
    """Root endpoint - API status"""
    return HealthResponse(
        status="running",
        model_loaded=vital_signs_predictor is not None,
        message="Vital Signs Prediction API is running"
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.head("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    is_healthy = vital_signs_predictor is not None
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=is_healthy,
        message="All systems operational" if is_healthy else "Model not loaded"
    )

@app.post("/predict", response_model=VitalSignsResponse, tags=["Prediction"])
async def predict_vital_signs(request: VitalSignsRequest, background_tasks: BackgroundTasks):
    """
    Predict vital signs from video URL
    
    - **video_url**: Direct URL to video file (mp4, avi, mov, webm)
    - **age**: Person's age (1-120, default: 25)
    - **gender**: Person's gender (M/F, default: M)
    """
    if not vital_signs_predictor:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable: Predictor not initialized"
        )
    
    temp_file_path = None
    try:
        logger.info(f"Processing prediction request for URL: {request.video_url[:50]}...")
        
        # Validate video URL
        if not request.video_url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid video URL format")
        
        # Download video
        temp_file_path = await video_processor.download_video(request.video_url)
        
        # Predict vitals
        import time
        start_time = time.time()
        logger.info(f"Starting prediction with age={request.age}, gender={request.gender}")
        
        try:
            results = vital_signs_predictor.predict_from_video(
                temp_file_path,
                request.age,
                request.gender
            )
        except Exception as pred_error:
            logger.error(f"Prediction error: {str(pred_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(pred_error)}"
            )
        
        processing_time = round(time.time() - start_time, 2)
        results['processing_time'] = processing_time
        
        logger.info(f"Prediction completed in {processing_time}s")
        
        # Schedule cleanup
        background_tasks.add_task(video_processor.cleanup_temp_file, temp_file_path)
        
        return VitalSignsResponse(**results)
    
    except HTTPException:
        if temp_file_path:
            background_tasks.add_task(video_processor.cleanup_temp_file, temp_file_path)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if temp_file_path:
            background_tasks.add_task(video_processor.cleanup_temp_file, temp_file_path)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/status", tags=["Status"])
async def get_status():
    """Get API status and capabilities"""
    if not vital_signs_predictor:
        return {
            "status": "Model not loaded",
            "error": "Predictor initialization failed",
            "model_loaded": False
        }
    return {
        "status": "Ready",
        "model_loaded": True,
        "supported_formats": ["mp4", "avi", "mov", "webm"],
        "max_video_size": "100MB (recommended)",
        "min_frames_required": 30,
        "api_version": "1.0.0",
        "memory_usage": "Optimized for cloud deployment"
    }

@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle preflight OPTIONS requests"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/test", tags=["Debug"])
async def test_endpoint():
    """Test endpoint for debugging"""
    return {
        "message": "Test endpoint working",
        "predictor_status": vital_signs_predictor is not None,
        "environment": {
            "PORT": os.environ.get("PORT", "Not set"),
            "python_version": sys.version.split()[0],
        }
    }

# Add a simple ping endpoint that responds immediately
@app.get("/ping", tags=["Health"])
async def ping():
    """Simple ping endpoint for quick health checks"""
    return {"message": "pong", "timestamp": __import__('time').time()}

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,  # Keep connections alive longer
    )