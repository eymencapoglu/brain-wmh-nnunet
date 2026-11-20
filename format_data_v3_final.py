import os
import shutil
import json
import glob
from pathlib import Path

# --- AYARLAR ---
ROOT_DIR = r"C:\Users\Eymen\Desktop\BEYİN PROJE\WMH\wmh_data"

# nnU-Net Hedef Klasörü
BASE_DIR = r"C:\Users\Eymen\Desktop\nnUNet_Project\nnUNet_raw"
DATASET_NAME = "Dataset001_WMH" 

TASK_DIR = os.path.join(BASE_DIR, DATASET_NAME)
IMAGES_TR = os.path.join(TASK_DIR, "imagesTr")
LABELS_TR = os.path.join(TASK_DIR, "labelsTr")

os.makedirs(IMAGES_TR, exist_ok=True)
os.makedirs(LABELS_TR, exist_ok=True)

print("--- Akıllı Veri Dönüştürme Başlıyor (V5 - Registered T1 FIX) ---")

# 1. ADIM: Maskeleri Bul (Aynı kaldı)
mask_base_dir = Path(ROOT_DIR) / "additional_annotations" / "observer_o3" / "training"
valid_cases = []

if not mask_base_dir.exists():
    print(f"HATA: Maske klasörü bulunamadı: {mask_base_dir}")
    exit()

for site_folder in mask_base_dir.iterdir():
    if site_folder.is_dir():
        site_name = site_folder.name
        for patient_folder in site_folder.iterdir():
            if patient_folder.is_dir():
                patient_id = patient_folder.name
                mask_file = patient_folder / "result.nii"
                if mask_file.exists():
                    valid_cases.append((site_name, patient_id, mask_file))

print(f"Bulunan Maske Sayısı: {len(valid_cases)}")

# 2. ADIM: Görüntüleri 'pre' Klasöründe Daha Spesifik Dosya İsimleriyle Ara
case_count = 0

for site, patient_id, mask_path in valid_cases:
    
    # Hasta ID'sine sahip klasörü bulma (training/test içinde)
    search_train = glob.glob(os.path.join(ROOT_DIR, "**", "training", site, "**", patient_id), recursive=True)
    search_test = glob.glob(os.path.join(ROOT_DIR, "**", "test", site, "**", patient_id), recursive=True)
    candidate_folders = search_train + search_test
    
    found_flair = None
    found_t1_reg = None
    
    for folder in candidate_folders:
        folder_path = Path(folder)
        pre_folder = folder_path / "pre"
        
        if not pre_folder.exists():
            continue
            
        # Kanal 0: FLAIR (Referans boyut)
        flair_c = pre_folder / "FLAIR.nii"
        if not flair_c.exists(): flair_c = pre_folder / "flair.nii"
        
        # Kanal 1: Registered T1 (Aynı boyutta olmalı)
        # WMH yarışmasında en yaygın kullanılan isimleri deniyoruz
        t1_reg_candidates = [
            pre_folder / "T1_reg.nii",  
            pre_folder / "T1_to_FLAIR.nii",
            pre_folder / "3DT1_reg.nii",
            pre_folder / "3DT1_to_FLAIR.nii",
            # Hatalı olsa bile önceki denemede kopyaladığımız isimler (genellikle boyutları farklı olanlar):
            pre_folder / "T1.nii",       
            pre_folder / "3DT1.nii"      
        ]
        
        if flair_c.exists():
            found_flair = flair_c
            for t1_cand in t1_reg_candidates:
                if t1_cand.exists():
                    found_t1_reg = t1_cand
                    break
        
        if found_flair and found_t1_reg:
            break # Dosyalar bulundu, iç döngüden çık
    
    # 3. ADIM: Kopyalama
    if found_flair and found_t1_reg:
        unique_id = f"{site}_{patient_id}"
        
        dst_flair = os.path.join(IMAGES_TR, f"{unique_id}_0000.nii")
        dst_t1 = os.path.join(IMAGES_TR, f"{unique_id}_0001.nii")
        dst_label = os.path.join(LABELS_TR, f"{unique_id}.nii")
        
        try:
            shutil.copy(str(found_flair), dst_flair)
            shutil.copy(str(found_t1_reg), dst_t1)
            shutil.copy(str(mask_path), dst_label)
            case_count += 1
            if case_count % 5 == 0:
                print(f"-> {case_count} vaka (T1 Registered) işlendi. (Son: {unique_id})")
        except Exception as e:
            print(f"HATA Kopyalama {unique_id}: {e}")
    else:
        print(f"UYARI: Görüntü (özellikle Registered T1) 'pre' klasöründe bulunamadı -> {site} {patient_id}")

print(f"\nİşlem Bitti. Toplam {case_count} vaka hazırlandı.")

# --- JSON OLUŞTURMA ---
if case_count > 0:
    json_dict = {
        "channel_names": {
            "0": "FLAIR",
            "1": "T1"
        },
        "labels": {
            "background": 0,
            "WMH": 1
        },
        "numTraining": case_count,
        "file_ending": ".nii"
    }

    with open(os.path.join(TASK_DIR, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4)
    print("dataset.json güncellendi. Eğitime geçmeye hazırız!")
else:
    print("HATA: Hiçbir veri kopyalanmadı.")