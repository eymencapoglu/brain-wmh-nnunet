<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WMH Yapay Zeka Analizi</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; display: flex; flex-direction: column; align-items: center; min-height: 100vh; margin: 0; padding: 20px; }
        h1 { color: #2c3e50; margin-bottom: 5px; }
        p.subtitle { color: #7f8c8d; margin-bottom: 30px; font-size: 0.9rem; }
        
        .container { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%; max-width: 800px; text-align: center; }
        
        .upload-area { border: 2px dashed #bdc3c7; border-radius: 8px; padding: 40px; cursor: pointer; transition: all 0.3s ease; position: relative; }
        .upload-area:hover, .upload-area.dragover { border-color: #3498db; background-color: #ecf0f1; }
        .upload-icon { font-size: 48px; color: #95a5a6; margin-bottom: 10px; }
        input[type="file"] { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
        
        .btn { background-color: #3498db; color: white; border: none; padding: 12px 30px; border-radius: 6px; font-size: 1rem; cursor: pointer; margin-top: 20px; transition: background 0.3s; display: none; }
        .btn:hover { background-color: #2980b9; }
        .btn:disabled { background-color: #bdc3c7; cursor: not-allowed; }
        
        .preview-section { margin-top: 30px; display: none; justify-content: center; gap: 20px; flex-wrap: wrap; }
        .image-card { flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; position: relative; }
        .image-card img { width: 100%; height: auto; display: block; }
        .image-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0.6; pointer-events: none; mix-blend-mode: multiply; }
        .card-header { background: #f8f9fa; padding: 10px; font-weight: bold; border-bottom: 1px solid #ddd; }
        
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto; display: none; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }1
        
        .error-msg { color: #e74c3c; margin-top: 15px; font-weight: bold; display: none; }
    </style>
</head>
<body>

    <h1>🧠 WMH Lezyon Tespit Sistemi</h1>
    <p class="subtitle">Yapay Zeka Destekli MR Analizi</p>

    <div class="container">
        <!-- Yükleme Alanı -->
        <div class="upload-area" id="drop-zone">
            <div class="upload-icon">📁</div>
            <p>MR Görüntüsünü (PNG/JPG) buraya sürükleyin veya tıklayın</p>
            <input type="file" id="file-input" accept="image/*">
        </div>
        <p id="file-name" style="margin-top: 10px; font-weight: bold; color: #2c3e50;"></p>
        
        <button class="btn" id="analyze-btn">Analiz Et</button>
        
        <div class="loader" id="loader"></div>
        <p class="error-msg" id="error-text"></p>
        
        <!-- Sonuç Alanı -->
        <div class="preview-section" id="results">
            <!-- Orijinal -->
            <div class="image-card">
                <div class="card-header">Orijinal Görüntü</div>
                <img id="original-img" src="" alt="Orijinal">
                <!-- Maske Overlay -->
                <!-- mix-blend-mode ve filter ile sadece kırmızı lezyonları gösteriyoruz -->
                <img id="mask-overlay" class="image-overlay" src="" alt="Lezyon" style="display:none; filter: drop-shadow(0 0 2px red);">
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('file-input');
        const analyzeBtn = document.getElementById('analyze-btn');
        const loader = document.getElementById('loader');
        const resultsDiv = document.getElementById('results');
        const originalImg = document.getElementById('original-img');
        const maskOverlay = document.getElementById('mask-overlay');
        const errorText = document.getElementById('error-text');
        const fileNameDisplay = document.getElementById('file-name');
        
        const API_URL = 'http://localhost:8001/predict';

        // 1. Dosya Seçimi
        fileInput.addEventListener('change', function(e) {
            if (this.files && this.files[0]) {
                const file = this.files[0];
                fileNameDisplay.textContent = "Seçilen Dosya: " + file.name;
                analyzeBtn.style.display = "inline-block";
                
                // Önizleme
                const reader = new FileReader();
                reader.onload = function(e) {
                    originalImg.src = e.target.result;
                    resultsDiv.style.display = "flex";
                    maskOverlay.style.display = "none"; 
                }
                reader.readAsDataURL(file);
            }
        });

        // 2. Analiz Butonu
        analyzeBtn.addEventListener('click', async () => {
             const file = fileInput.files[0];
             if (!file) return;
             
             // UI Hazırla
             analyzeBtn.disabled = true;
             analyzeBtn.textContent = "İşleniyor...";
             loader.style.display = "block";
             errorText.style.display = "none";
             maskOverlay.style.display = "none";
             
             const formData = new FormData();
             formData.append('file', file);
             
             try {
                 const response = await fetch(API_URL, {
                     method: 'POST',
                     body: formData
                 });

                 // Raw text'i alalım ki hata HTML mi görelim
                 const responseText = await response.text();

                 if (!response.ok) {
                     console.error("Sunucu Hatası:", responseText);
                     throw new Error(`Sunucu Hatası (${response.status}): ${responseText.substring(0, 100)}...`);
                 }

                 let data;
                 try {
                     data = JSON.parse(responseText);
                 } catch (e) {
                     console.error("JSON Parse Hatası. Gelen Yanıt:", responseText);
                     // Hata HTML ise kullanıcıya gösterelim
                     if (responseText.trim().startsWith("<")) {
                         // HTML içeriğini yeni pencerede açabiliriz veya sadece uyarı veririz
                         throw new Error("Sunucu JSON yerine HTML döndürdü. (Bkz: Konsol)");
                     }
                     throw new Error("Geçersiz Yanıt Formatı");
                 }
                 
                 if (data.status === "success") {
                     // Başarılı
                     const maskBase64 = "data:image/png;base64," + data.mask_base64;
                     
                     maskOverlay.src = maskBase64;
                     
                     // Overlay Stili
                     maskOverlay.style.display = "block";
                     // Lezyonları belirginleştirmek için stil ayarları
                     maskOverlay.style.filter = "sepia(1) saturate(10) hue-rotate(-50deg) opacity(0.8)";
                     maskOverlay.style.mixBlendMode = "normal"; 
                     
                 } else {
                     throw new Error("API işlem yapamadı.");
                 }
                 
             } catch (err) {
                 console.error(err);
                 errorText.textContent = "Hata oluştu: " + err.message;
                 errorText.style.display = "block";
             } finally {
                 loader.style.display = "none";
                 analyzeBtn.disabled = false;
                 analyzeBtn.textContent = "Tekrar Analiz Et";
             }
        });
    </script>
</body>
</html>
