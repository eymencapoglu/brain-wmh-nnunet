import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# --- AYARLAR ---
# Klasör yollarını tekrar kontrol ederek yazıyoruz
IMAGES_DIR = r"C:\Users\Eymen\Desktop\nnUNet_Project\nnUNet_raw\Dataset001_WMH\imagesTs"
PRED_DIR = r"C:\Users\Eymen\Desktop\nnUNet_Project\tahminler"
OUTPUT_DIR = r"C:\Users\Eymen\Desktop\nnUNet_Project\gorsel_sonuclar"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- HATA AYIKLAMA VE GÖRSELLEŞTİRME ---")
print(f"Tahmin Klasörü: {PRED_DIR}")

# 1. Klasör Kontrolü
if not os.path.exists(PRED_DIR):
    print(f"HATA: '{PRED_DIR}' klasörü bulunamadı!")
    print("Lütfen tahmin klasörünün ismini kontrol edin.")
    exit()

# 2. Dosya Listeleme (Ne var ne yok bakalım)
all_files = os.listdir(PRED_DIR)
print(f"Klasördeki toplam dosya sayısı: {len(all_files)}")

if len(all_files) == 0:
    print("HATA: Tahmin klasörü BOŞ! Tahmin (prediction) işlemi dosyaları kaydetmemiş.")
    print("Çözüm: nnUNetv2_predict komutunu tekrar çalıştırın.")
    exit()
else:
    print(f"İlk 3 dosya örneği: {all_files[:3]}")

# 3. Uzantıya Göre Arama (.nii veya .nii.gz)
pred_files = glob.glob(os.path.join(PRED_DIR, "*.nii.gz"))
if not pred_files:
    print("Uyarı: .nii.gz dosyası bulunamadı, .nii aranıyor...")
    pred_files = glob.glob(os.path.join(PRED_DIR, "*.nii"))

print(f"İşlenecek Maske Sayısı: {len(pred_files)}")

if not pred_files:
    print("HATA: Ne .nii ne de .nii.gz dosyası bulundu. Klasörde başka türde dosyalar olabilir.")
    exit()

# 4. Görselleştirme Döngüsü
print("\n--- Görsel Oluşturma Başlıyor ---")

for i, pred_path in enumerate(pred_files):
    filename = os.path.basename(pred_path)
    # Uzantıyı temizle (.nii.gz veya .nii)
    case_id = filename.replace(".nii.gz", "").replace(".nii", "")
    
    # Orijinal FLAIR dosyasını bul (nnU-Net formatında _0000.nii)
    flair_path = os.path.join(IMAGES_DIR, f"{case_id}_0000.nii")
    
    if not os.path.exists(flair_path):
        # Belki .nii.gz dir?
        flair_path = os.path.join(IMAGES_DIR, f"{case_id}_0000.nii.gz")
        if not os.path.exists(flair_path):
            print(f"UYARI: Orijinal görüntü bulunamadı -> {case_id}")
            continue
        
    try:
        # Dosyaları Yükle
        img_nii = nib.load(flair_path)
        mask_nii = nib.load(pred_path)
        
        img_data = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        
        # En Çok Lezyon Olan Kesiti Bul
        if np.sum(mask_data) > 0:
            # 3 eksende de toplayıp en yoğun olanı bulalım (genelde eksen 2 derinliktir)
            # Ama bazen oryantasyon farklı olabilir, basitçe en çok piksel olan slice'ı alalım.
            slice_sums = np.sum(mask_data, axis=(0, 1))
            best_slice_idx = np.argmax(slice_sums)
            has_lesion = True
        else:
            best_slice_idx = mask_data.shape[2] // 2
            has_lesion = False
            
        # Görüntüyü Hazırla (90 derece döndürerek düzeltme)
        slice_img = np.rot90(img_data[:, :, best_slice_idx])
        slice_mask = np.rot90(mask_data[:, :, best_slice_idx])
        
        # Çizim
        plt.figure(figsize=(10, 5))
        
        # Sol: Orijinal
        plt.subplot(1, 2, 1)
        plt.imshow(slice_img, cmap='gray')
        plt.title(f"{case_id}\nOrijinal (Kesit: {best_slice_idx})")
        plt.axis('off')
        
        # Sağ: Tahmin
        plt.subplot(1, 2, 2)
        plt.imshow(slice_img, cmap='gray')
        if has_lesion:
            masked_overlay = np.ma.masked_where(slice_mask == 0, slice_mask)
            plt.imshow(masked_overlay, cmap='autumn', alpha=0.7)
            plt.title("Yapay Zeka Tespiti (Lezyon Var)")
        else:
            plt.title("Yapay Zeka Tespiti (Temiz)")
        plt.axis('off')
        
        save_path = os.path.join(OUTPUT_DIR, f"{case_id}_gorsel.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Kaydedildi: {case_id}_gorsel.png")
        
        # İlk 10 taneden sonra duralım
        if i >= 10: 
            break
            
    except Exception as e:
        print(f"HATA {case_id}: {e}")

print(f"\nİşlem Tamamlandı! Klasör: {OUTPUT_DIR}")