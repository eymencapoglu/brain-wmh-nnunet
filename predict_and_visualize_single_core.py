import os
import subprocess
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

# ==============================================================================
# --- AYARLAR (KENDİ BİLGİSAYARINIZA GÖRE) ---
# ==============================================================================

# 1. Ana Klasör Yolları
PROJECT_ROOT = r"C:\Users\Eymen\Desktop\nnUNet_Project"
NNUNET_RAW = os.path.join(PROJECT_ROOT, "nnUNet_raw")
NNUNET_PREPROCESSED = os.path.join(PROJECT_ROOT, "nnUNet_preprocessed")
NNUNET_RESULTS = os.path.join(PROJECT_ROOT, "nnUNet_results")

DATASET_NAME = "Dataset001_WMH"

# 2. Girdi ve Çıktı Klasörleri
INPUT_IMAGES_DIR = os.path.join(NNUNET_RAW, DATASET_NAME, "imagesTs")
TEMP_3D_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "temp_3d_predictions")
FINAL_PNG_DIR = os.path.join(PROJECT_ROOT, "FINAL_PNG_SONUCLAR")

# 3. Model Dosyası
CHECKPOINT_NAME = "checkpoint_best.pth"

# ==============================================================================
# --- HATA DÜZELTME: ORTAM DEĞİŞKENLERİNİ AYARLAMA ---
# ==============================================================================
# Terminalde hata almamak için yolları Python içinden tanımlıyoruz.
os.environ["nnUNet_raw"] = NNUNET_RAW
os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
os.environ["nnUNet_results"] = NNUNET_RESULTS

# ==============================================================================
# --- ADIM 1: nnU-Net İLE 3D TAHMİN YAPMA ---
# ==============================================================================
print("\n" + "="*50)
print("ADIM 1/2: 3D Tahmin Başlıyor...")
print("(Bu işlem GPU/CPU gücüne göre birkaç dakika sürebilir)")
print("="*50)

# Klasörleri temizle ve oluştur
if os.path.exists(TEMP_3D_OUTPUT_DIR): shutil.rmtree(TEMP_3D_OUTPUT_DIR)
os.makedirs(TEMP_3D_OUTPUT_DIR, exist_ok=True)

if os.path.exists(FINAL_PNG_DIR): shutil.rmtree(FINAL_PNG_DIR)
os.makedirs(FINAL_PNG_DIR, exist_ok=True)

# Tahmin Komutu
command = [
    "nnUNetv2_predict",
    "-i", INPUT_IMAGES_DIR,
    "-o", TEMP_3D_OUTPUT_DIR,
    "-d", "1",
    "-c", "3d_fullres",
    "-f", "0",
    "-chk", CHECKPOINT_NAME
]

try:
    # Komutu çalıştır ve ortam değişkenlerini aktar
    subprocess.check_call(command, shell=True, env=os.environ)
    print("✅ 3D Tahmin işlemi başarıyla tamamlandı.")
except subprocess.CalledProcessError as e:
    print(f"\n❌ HATA: nnU-Net çalışırken hata oluştu. Kod: {e.returncode}")
    exit()

# ==============================================================================
# --- ADIM 2: PNG GÖRSELLERİ OLUŞTURMA ---
# ==============================================================================
print("\n" + "="*50)
print("ADIM 2/2: PNG Görselleri Oluşturuluyor...")
print("="*50)

pred_files = glob.glob(os.path.join(TEMP_3D_OUTPUT_DIR, "*.nii.gz"))

if not pred_files:
    print("HATA: Tahmin klasörü boş! Adım 1 başarısız olmuş.")
    exit()

for i, pred_path in enumerate(pred_files):
    try:
        filename = os.path.basename(pred_path)
        case_id = filename.replace(".nii.gz", "")
        
        # Orijinal FLAIR dosyasını bul
        flair_path = os.path.join(INPUT_IMAGES_DIR, f"{case_id}_0000.nii")
        
        # Yükle
        img_nii = nib.load(flair_path)
        mask_nii = nib.load(pred_path)
        img_data = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        
        # En yoğun kesiti bul
        if np.sum(mask_data) > 0:
            slice_sums = np.sum(mask_data, axis=(0, 1))
            best_slice_idx = np.argmax(slice_sums)
            has_lesion = True
            status_text = "Tespit: VAR"
        else:
            best_slice_idx = mask_data.shape[2] // 2
            has_lesion = False
            status_text = "Tespit: YOK"

        # Görüntüyü hazırla (Döndür)
        slice_img = np.rot90(img_data[:, :, best_slice_idx])
        slice_mask = np.rot90(mask_data[:, :, best_slice_idx])
        
        # Çizim
        plt.figure(figsize=(12, 6))
        
        # Sol: Ham Görüntü
        plt.subplot(1, 2, 1)
        plt.imshow(slice_img, cmap='gray')
        plt.title(f"{case_id}\nOrijinal (Kesit: {best_slice_idx})")
        plt.axis('off')
        
        # Sağ: Tahminli Görüntü
        plt.subplot(1, 2, 2)
        plt.imshow(slice_img, cmap='gray')
        
        if has_lesion:
            masked_overlay = np.ma.masked_where(slice_mask == 0, slice_mask)
            plt.imshow(masked_overlay, cmap='autumn', alpha=0.7)
        
        plt.title(f"Yapay Zeka Sonucu ({status_text})")
        plt.axis('off')
        
        # Kaydet
        save_path = os.path.join(FINAL_PNG_DIR, f"{case_id}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"[{i+1}/{len(pred_files)}] İşlendi: {case_id}")
        
    except Exception as e:
        print(f"HATA {case_id}: {e}")

print("\n" + "="*50)
print(f"✅ İŞLEM BİTTİ! Görselleriniz şurada hazır:")
print(f"{FINAL_PNG_DIR}")
print("="*50)