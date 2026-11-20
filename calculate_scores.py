import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path

# --- AYARLAR ---
# Kaggle verisinin olduğu yer
ROOT_DIR = r"C:\Users\Eymen\Desktop\BEYİN PROJE\WMH\wmh_data"
# Modelin ürettiği tahminlerin olduğu yer
PRED_DIR = r"C:\Users\Eymen\Desktop\nnUNet_Project\tahminler"

def dice_score(pred, target):
    # Basit Dice katsayısı hesaplama
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

print("--- Skor Hesaplama ve Kontrol ---")

pred_files = glob.glob(os.path.join(PRED_DIR, "*.nii.gz"))
print(f"Toplam Tahmin Sayısı: {len(pred_files)}")

scores = []

for pred_path in pred_files:
    filename = os.path.basename(pred_path)
    # Dosya adı: Amsterdam_111.nii.gz -> Site: Amsterdam, ID: 111
    parts = filename.replace(".nii.gz", "").split("_")
    
    if len(parts) >= 2:
        site = parts[0]
        p_id = parts[1]
        
        # Orijinal Maskeyi Ara (wmh.nii veya result.nii)
        # Kaggle yapısında test klasöründe genellikle maske olmaz ama kontrol edelim
        # Olası yerler: test/Site/Scanner/ID/wmh.nii
        
        # Test klasörünü tara
        search_pattern = os.path.join(ROOT_DIR, "**", "test", site, "**", p_id, "wmh.nii")
        candidates = glob.glob(search_pattern, recursive=True)
        
        # wmh.nii bulamazsa result.nii ara
        if not candidates:
            search_pattern = os.path.join(ROOT_DIR, "**", "test", site, "**", p_id, "result.nii")
            candidates = glob.glob(search_pattern, recursive=True)

        if candidates:
            mask_path = candidates[0]
            
            # Maske ve Tahmini Yükle
            pred_img = nib.load(pred_path).get_fdata()
            mask_img = nib.load(mask_path).get_fdata()
            
            # Boyut kontrolü (Bazen maske hizalanmamış olabilir)
            if pred_img.shape != mask_img.shape:
                print(f"⚠️ Boyut Uyumsuzluğu: {filename} (Atlanıyor)")
                continue
                
            # Skor Hesapla
            dice = dice_score(pred_img, mask_img)
            scores.append(dice)
            print(f"✅ {filename}: Dice = {dice:.4f}")
        else:
            # Eğer maske bulunamazsa
            print(f"❌ {filename}: Cevap anahtarı (Maske) bulunamadı.")

if scores:
    print(f"\n--- SONUÇLAR ---")
    print(f"Ortalama Dice Skoru: {np.mean(scores):.4f}")
else:
    print("\n--- SONUÇ ---")
    print("Test verisi için yerel maske (Ground Truth) bulunamadı.")
    print("Bu yüzden sayısal bir başarı skoru üretilemiyor.")
    print("Lütfen sonuçları ITK-SNAP ile görsel olarak analiz edin.")