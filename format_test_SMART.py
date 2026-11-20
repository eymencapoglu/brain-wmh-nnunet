import os
import shutil
import glob
import nibabel as nib
from pathlib import Path

# --- AYARLAR ---
ROOT_DIR = r"C:\Users\Eymen\Desktop\BEYİN PROJE\WMH\wmh_data"
BASE_DIR = r"C:\Users\Eymen\Desktop\nnUNet_Project\nnUNet_raw"
DATASET_NAME = "Dataset001_WMH" 

TASK_DIR = os.path.join(BASE_DIR, DATASET_NAME)
IMAGES_TS = os.path.join(TASK_DIR, "imagesTs")

# Klasörü temizleyip yeniden oluşturalım
if os.path.exists(IMAGES_TS):
    shutil.rmtree(IMAGES_TS)
os.makedirs(IMAGES_TS, exist_ok=True)

print("--- Akıllı Test Verisi Hazırlama (BOYUT KONTROLLÜ) ---")
print(f"Hedef: {IMAGES_TS}")

test_base_dir = Path(ROOT_DIR) / "test"
if not test_base_dir.exists():
    test_base_dir = Path(ROOT_DIR) / "wmh_data" / "test"

valid_count = 0
mismatch_count = 0

if not test_base_dir.exists():
    print("HATA: Test klasörü bulunamadı!")
else:
    for site_folder in test_base_dir.iterdir():
        if site_folder.is_dir():
            site_name = site_folder.name
            for scanner_folder in site_folder.iterdir():
                if scanner_folder.is_dir():
                    for patient_folder in scanner_folder.iterdir():
                        if patient_folder.is_dir():
                            patient_id = patient_folder.name
                            pre_folder = patient_folder / "pre"
                            
                            if not pre_folder.exists(): continue
                                
                            # 1. Dosyaları Bul (FLAIR ve T1 adayları)
                            flair_path = pre_folder / "FLAIR.nii"
                            if not flair_path.exists(): flair_path = pre_folder / "flair.nii"
                            
                            # T1 adayları
                            t1_candidates = ["T1.nii", "3DT1.nii", "T1_to_FLAIR.nii", "3DT1_to_FLAIR.nii"]
                            t1_path = None
                            
                            # FLAIR yoksa geç
                            if not flair_path.exists(): continue

                            # FLAIR Boyutunu Öğren
                            try:
                                flair_img = nib.load(str(flair_path))
                                flair_shape = flair_img.shape
                            except:
                                print(f"HATA: FLAIR okunamadı {patient_id}")
                                continue

                            # T1 Adaylarını Tara ve Boyutu Eşleşeni Bul
                            for cand in t1_candidates:
                                temp_t1 = pre_folder / cand
                                if temp_t1.exists():
                                    try:
                                        t1_img = nib.load(str(temp_t1))
                                        if t1_img.shape == flair_shape:
                                            t1_path = temp_t1
                                            break # Eşleşen bulduk!
                                    except:
                                        continue
                            
                            unique_id = f"{site_name}_{patient_id}"

                            if t1_path:
                                # Eşleşme var, kopyala
                                dst_flair = os.path.join(IMAGES_TS, f"{unique_id}_0000.nii")
                                dst_t1 = os.path.join(IMAGES_TS, f"{unique_id}_0001.nii")
                                
                                shutil.copy(str(flair_path), dst_flair)
                                shutil.copy(str(t1_path), dst_t1)
                                valid_count += 1
                                print(f"✅ EKLENDİ: {unique_id} (Shape: {flair_shape})")
                            else:
                                mismatch_count += 1
                                print(f"❌ ATLANDI (Boyut Uyumsuz): {unique_id} - FLAIR: {flair_shape}")

print(f"\n--- ÖZET ---")
print(f"Toplam Eklenen (Uyumlu): {valid_count}")
print(f"Toplam Atlanan (Uyumsuz): {mismatch_count}")

if valid_count == 0:
    print("\nUYARI: Hiçbir dosya eklenemedi! Test setindeki hiçbir T1 dosyası FLAIR ile aynı boyutta değil.")
    print("Bu durumda bu test setini ham haliyle nnU-Net ile kullanamazsınız (Registration gerekir).")