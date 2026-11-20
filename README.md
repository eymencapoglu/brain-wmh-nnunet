# Brain MRI White Matter Hyperintensity (WMH) Segmentation with nnU-Net v2 + FastAPI Web UI

*[English](#english) | [Türkçe](#türkçe)*

---

<a name="english"></a>
## English

### Overview
This project is an end-to-end **3D Brain MRI White Matter Hyperintensity (WMH) Segmentation** pipeline built with **nnU-Net v2** (`Dataset001_WMH`, `3d_fullres`) on the **MICCAI 2017 WMH Segmentation Challenge** multi-center dataset (UMC Utrecht, NUHS Singapore, VUmc Amsterdam). Beyond dataset formatting, training, Dice evaluation, and slice visualization, the project includes a **FastAPI inference server (`api/`)** and a **drag-and-drop web interface (`web/index.php`)** that accepts standard 2D MRI PNG/JPG slices, converts them into dual-channel NIfTI volumes (FLAIR + T1), runs `nnUNetv2_predict`, and returns the predicted lesion mask as Base64 PNG.

### Development Timeline & Process
- **December 1, 2025:** Developed `format_data_v3_final.py` to convert the multi-center MICCAI 2017 WMH dataset (`BEYİN PROJE/WMH/wmh_data`, registered T1 + FLAIR + observer masks) into nnU-Net v2's `Dataset001_WMH` format.
- **December 4–5, 2025:** Built `format_test_SMART.py` (dimension-checked test preparation), `calculate_scores.py` (Dice coefficient evaluation), `visualize_debug.py`, and `visualize_results.py`.
- **December 21, 2025:** Added `predict_and_visualize_single_core.py` for single-core CPU inference and PNG export.
- **January 9–10, 2026:** Built the production-style local inference stack: FastAPI backend (`api/main.py`, `api/utils.py`), launcher (`start_api.bat`), and web frontend (`web/index.php`).

### Project Structure
```text
brain-wmh-nnunet/
├── api/
│   ├── main.py                              # FastAPI server wrapping nnUNetv2_predict
│   └── utils.py                             # PNG <-> NIfTI conversion & Base64 mask export
├── web/
│   └── index.php                            # Drag-and-drop MRI WMH web UI
├── format_data_v3_final.py                  # MICCAI WMH -> nnU-Net v2 training formatter
├── format_test_SMART.py                     # Shape-validated test formatter
├── calculate_scores.py                      # Dice score evaluator
├── predict_and_visualize_single_core.py     # Single-core CPU prediction & PNG export
├── visualize_debug.py / visualize_results.py
└── start_api.bat                            # Starts FastAPI on port 8001 + opens web UI
```

### Setup & Running
Multi-gigabyte NIfTI MRI volumes (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) and trained checkpoints are excluded via `.gitignore`.
1. Place your trained `nnUNet_results/Dataset001_WMH/.../checkpoint_best.pth` in the project directory (or set `NNUNET_PROJECT_ROOT` per `.env.example`).
2. Install dependencies and launch the API:
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8001
```
3. Open `web/index.php` via a local PHP server or run `start_api.bat` on Windows.

### Acknowledgments & Open-Source Credits
We deeply appreciate the open-source neuroimaging frameworks and datasets that powered this work:
- **[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)** — Isensee et al., *Nature Methods* (2021), German Cancer Research Center (DKFZ).
- **[MICCAI 2017 WMH Segmentation Challenge Dataset](https://wmh.isi.uu.nl/)** — Kuijf et al., UMC Utrecht, NUHS Singapore, and VUmc Amsterdam.
- **[NiBabel](https://nipy.org/nibabel/)**, **[FastAPI](https://fastapi.tiangolo.com/)**, **[PyTorch](https://pytorch.org/)**, and **[Pillow](https://python-pillow.org/)**.

---

<a name="türkçe"></a>
## Türkçe

### Genel Bakış
Bu proje, **MICCAI 2017 WMH Segmentation Challenge** çok merkezli beyin MR veri seti (Utrecht, Singapore, Amsterdam) üzerinde **nnU-Net v2** (`Dataset001_WMH`, `3d_fullres`) kullanarak geliştirilmiş uçtan uca **3B Beyin MR Beyaz Cevher Hiperintensitesi (WMH) Segmentasyon** sistemidir. Veri formatlama, Dice skorlama ve görselleştirme betiklerinin yanı sıra; kullanıcının yüklediği 2B MR görüntüsünü çift kanallı NIfTI hacmine dönüştürüp `nnUNetv2_predict` ile tahmin alan **FastAPI servisini (`api/`)** ve sürükle-bırak destekli **web arayüzünü (`web/index.php`)** içerir.

### Geliştirme Süreci ve Tarihçesi
- **1 Aralık 2025:** MICCAI 2017 WMH veri setindeki FLAIR, T1 ve uzman maskelerini nnU-Net v2 formatına (`Dataset001_WMH`) dönüştüren `format_data_v3_final.py` yazıldı.
- **4–5 Aralık 2025:** Boyut kontrollü test veri hazırlayıcısı (`format_test_SMART.py`), Dice skor hesaplayıcı (`calculate_scores.py`) ve görselleştirme betikleri eklendi.
- **21 Aralık 2025:** Tek çekirdekli CPU ortamları için `predict_and_visualize_single_core.py` geliştirildi.
- **9–10 Ocak 2026:** FastAPI tahmin servisi (`api/main.py`, `api/utils.py`), başlatıcı (`start_api.bat`) ve web arayüzü (`web/index.php`) tamamlandı.

### Kurulum ve Temizlik Notu
`api/main.py` içindeki sabit masaüstü yolu dinamik hale getirilmiş (`NNUNET_PROJECT_ROOT`), gigabaytlarca yer kaplayan ham NIfTI beyin MR hacimleri ve model ağırlıkları `.gitignore` ile hariç tutulmuştur.

### Açık Kaynak Teşekkürler ve Kaynaklar
- **[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)** — Isensee vd., *Nature Methods* (2021), DKFZ.
- **[MICCAI 2017 WMH Segmentation Challenge Veri Seti](https://wmh.isi.uu.nl/)** — Kuijf vd. (UMC Utrecht, NUHS Singapore, VUmc Amsterdam).
- **[NiBabel](https://nipy.org/nibabel/)**, **[FastAPI](https://fastapi.tiangolo.com/)** ve **[PyTorch](https://pytorch.org/)**.