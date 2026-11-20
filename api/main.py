from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import subprocess
import uuid
from typing import Dict
from api.utils import prepare_nifti_from_image, convert_nifti_to_base64_png

app = FastAPI()

# Enable CORS for PHP frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
PROJECT_ROOT = os.getenv("NNUNET_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NNUNET_RAW = os.path.join(PROJECT_ROOT, "nnUNet_raw")
NNUNET_PREPROCESSED = os.path.join(PROJECT_ROOT, "nnUNet_preprocessed")
NNUNET_RESULTS = os.path.join(PROJECT_ROOT, "nnUNet_results")

# Paths for inference
dataset_name = "Dataset001_WMH"
checkpoint_name = "checkpoint_best.pth"

# Set Evironment Variables for nnU-Net
os.environ["nnUNet_raw"] = NNUNET_RAW
os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
os.environ["nnUNet_results"] = NNUNET_RESULTS

# Temporary Directory for API processing
API_TEMP_DIR = os.path.join(PROJECT_ROOT, "api_temp_data")
os.makedirs(API_TEMP_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Setup Unique Job ID
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(API_TEMP_DIR, job_id)
    input_dir = os.path.join(job_dir, "input")
    output_dir = os.path.join(job_dir, "output")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 2. Read Image and Convert to NIfTI (Single -> Dual Channel)
        contents = await file.read()
        case_id = f"case_{job_id}"
        
        prepare_nifti_from_image(contents, case_id, input_dir)
        
        # 3. Run Inference
        # Command: nnUNetv2_predict -i INPUT -o OUTPUT -d 1 -c 3d_fullres -f 0 -chk checkpoint_best.pth
        command = [
            "nnUNetv2_predict",
            "-i", input_dir,
            "-o", output_dir,
            "-d", "1",
            "-c", "3d_fullres",
            "-f", "0",
            "-chk", checkpoint_name,
            "--disable_printing"
        ]
        
        # Run process
        process = subprocess.run(command, capture_output=True, env=os.environ, text=True, shell=True)
        
        if process.returncode != 0:
            print(f"STDERR: {process.stderr}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {process.stderr}")

        # 4. Process Output
        result_nii = os.path.join(output_dir, f"{case_id}.nii.gz")
        
        if not os.path.exists(result_nii):
             raise HTTPException(status_code=500, detail="Output file not created by model.")
             
        mask_b64 = convert_nifti_to_base64_png(result_nii)
        
        return {
            "status": "success",
            "job_id": job_id,
            "mask_base64": mask_b64
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        shutil.rmtree(job_dir, ignore_errors=True)

@app.get("/")
def home():
    return {"message": "nnU-Net Inference API Ready"}
