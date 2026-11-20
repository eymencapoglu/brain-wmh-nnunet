import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# --- AYARLAR ---
# Orijinal Test Görüntüleri (FLAIR olanlar)
IMAGES_DIR = r"C:\Users\Eymen\Desktop\nnUNet_Project\nnUNet_raw\Dataset001_WMH\imagesTs"
# Modelin Tahminleri
PRED_DIR = r"C:\Users\Eymen\Desktop\nnUNet_Project\tahminler"
# Çıktıların Kaydedileceği Klasör
OUTPUT_DIR = r"C:\Users\Eymen\Desktop\nnUNet_Project\gorsel_sonuclar"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Görselleştirme Başlıyor ---")

# Tahmin dosyalarını bul
pred_files = glob.glob(os.path.join(PRED_DIR, "*.nii.gz"))

for i, pred_path in enumerate(pred_files):
    filename = os.path.basename(pred_path)
    case_id = filename.replace(".nii.gz", "")
    
    # Orijinal FLAIR dosyasını bul (nnU-Net formatında _0000.nii)
    flair_path = os.path.join(IMAGES_DIR, f"{case_id}_0000.nii")
    
    if not os.path.exists(flair_path):
        print(f"UYARI: Orijinal görüntü bulunamadı -> {case_id}")
        continue
        
    try:
        # 1. Dosyaları Yükle
        img_nii = nib.load(flair_path)
        mask_nii = nib.load(pred_path)
        
        img_data = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        
        # 2. En Çok Lezyon Olan Kesiti Bul (Akıllı Seçim)
        # Maskenin dolu olduğu pikselleri say, en yoğun dilimi seç
        # Eksenler (0,1) genellikle yatay düzlemdir, 2. eksen derinliktir (dilimler)
        if np.sum(mask_data) > 0:
            slice_sums = np.sum(mask_data, axis=(0, 1))
            best_slice_idx = np.argmax(slice_sums)
        else:
            # Hiç lezyon bulamadıysa orta kesiti al
            best_slice_idx = mask_data.shape[2] // 2
            
        # 3. Görüntüyü Hazırla (Matplotlib için döndürmek gerekebilir)
        # 3D veriden 2D kesit alıyoruz
        slice_img = np.rot90(img_data[:, :, best_slice_idx])
        slice_mask = np.rot90(mask_data[:, :, best_slice_idx])
        
        # 4. Çizim Yap
        plt.figure(figsize=(12, 6))
        
        # Sol: Orijinal Görüntü
        plt.subplot(1, 2, 1)
        plt.imshow(slice_img, cmap='gray')
        plt.title(f"{case_id} - Orijinal (Kesit: {best_slice_idx})")
        plt.axis('off')
        
        # Sağ: Tahmin (Overlay)
        plt.subplot(1, 2, 2)
        plt.imshow(slice_img, cmap='gray')
        # Maskeyi yarı saydam (alpha=0.5) kırmızı olarak üstüne koy
        # Maske olmayan yerleri (0) gizle (np.nan yap)
        masked_overlay = np.ma.masked_where(slice_mask == 0, slice_mask)
        plt.imshow(masked_overlay, cmap='autumn', alpha=0.6) 
        plt.title(f"Yapay Zeka Tahmini (WMH)")
        plt.axis('off')
        
        # 5. Kaydet
        save_path = os.path.join(OUTPUT_DIR, f"{case_id}_analiz.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Kaydedildi: {case_id}_analiz.png")
        
        # İlk 10 tanesini yapıp duralım (Hepsini isterseniz bu break'i kaldırın)
        if i >= 50: 
            break
            
    except Exception as e:
        print(f"HATA {case_id}: {e}")

print(f"\nİşlem Bitti! '{OUTPUT_DIR}' klasörüne bakabilirsiniz.")