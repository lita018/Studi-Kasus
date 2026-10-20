# Deteksi Perubahan Tutupan Lahan Otomatis dengan NDVI Image Differencing

**Latar belakang:**
Pemateri bercerita tentang pemetaan lahan pertanian pangan berkelanjutan agar tidak terjadi alih fungsi lahan menjadi perumahan atau sawit, dan tentang klasifikasi lahan otomatis dengan remote sensing. Menggambar poligon manual dianggap tidak efisien.

**Rumusan masalah:** 
Bagaimana mendeteksi area yang berubah dari vegetasi menjadi lahan terbuka atau terbangun secara otomatis dari dua citra waktu berbeda?

**Metode:**
    1) Hitung NDVI untuk citra waktu 1 dan waktu 2 (band Red dan NIR).
    2) Hitung selisih NDVI. Penurunan di bawah ambang batas (default −0.25) dianggap kehilangan vegetasi.
    3) Bersihkan noise dengan morphological opening, lalu buang objek kecil.
    4) Beri label tiap area berubah, lalu hitung luas dan bounding box (mirip atribut data GIS).
    5)Simpan laporan JSON dan visualisasi.

**Cara pakai:**
  1) Mode demo (data sintetis, tanpa file apapun):
        python change_detection.py
  2) Mode data nyata (dua citra GeoTIFF multi-band, band Red & NIR):
        python change_detection.py --t1 citra_2020.tif --t2 citra_2025.tif --red 3 --nir 4

**Cara Menjalankan:**
'''bash
pip install scipy rasterio
conda install -c conda-forge scipy rasterio