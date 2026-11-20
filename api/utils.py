import os
import io
import numpy as np
import nibabel as nib
from PIL import Image
import base64

def prepare_nifti_from_image(image_bytes: bytes, case_id: str, output_dir: str):
    """
    Converts 2D image bytes to 3D NIfTI files (Channel 0 and Channel 1) required by nnU-Net.
    Since the user provides one image, we duplicate it for both channels.
    """
    # 1. Load Image
    image = Image.open(io.BytesIO(image_bytes)).convert('L') # Convert to Grayscale
    img_data = np.array(image)
    
    # 2. Reshape to 3D (H, W, 1) -> nnU-Net expects volumes
    # Rotation might be needed depending on how training data was fed, 
    # but starting with direct mapping is safest.
    # Note: Nibabel and standard viewing often swap X/Y compared to PIL.
    # We rotate 90 degrees to match typical medical orientation if needed, 
    # but let's stick to raw for now.
    img_data_3d = img_data[:, :, np.newaxis] 
    
    # Cast to float32 (standard for processing)
    img_data_3d = img_data_3d.astype(np.float32)

    # 3. Create NIfTI Object (Affine is identity for unreferenced PNGs)
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(img_data_3d, affine)

    # 4. Save Channels
    # Channel 0: FLAIR (Real input)
    c0_path = os.path.join(output_dir, f"{case_id}_0000.nii.gz")
    nib.save(nifti_img, c0_path)
    
    # Channel 1: T1 (Duplicated input)
    c1_path = os.path.join(output_dir, f"{case_id}_0001.nii.gz")
    nib.save(nifti_img, c1_path)
    
    return c0_path, c1_path

def convert_nifti_to_base64_png(nifti_path: str):
    """
    Reads a prediction NIfTI file, extracts the slice, and converts to Base64 PNG.
    """
    if not os.path.exists(nifti_path):
        return None

    # Load NIfTI
    nimg = nib.load(nifti_path)
    data = nimg.get_fdata()
    
    # Data is (H, W, 1), simple squeeze
    slice_data = np.squeeze(data)
    
    # Normalize or Threshold (Labels are 0 and 1)
    # If 1 (WMH), we want it visible.
    # Let's return just the raw mask as PNG, frontend handles overlay?
    # Or better: result is the MASK itself (black/white or transparent/red).
    
    # For simplicity, let's return a mask image where 1=255 (White), 0=0 (Black)
    mask_uint8 = (slice_data * 255).astype(np.uint8)
    
    res_img = Image.fromarray(mask_uint8, mode='L')
    
    # Convert to Base64
    buff = io.BytesIO()
    res_img.save(buff, format="PNG")
    b64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    
    return b64_str

def apply_overlay(original_bytes: bytes, mask_b64: str):
    """
    Optional: Merges original image and mask here if needed.
    """
    # Skipping for now, will send mask and let frontend/user see it or 
    # simple overlay logic can happen in main if requested.
    pass
