# HPC Image Forensics — Studi Kasus Deteksi Manipulasi Citra Skala Besar

Studi kasus dari kuliah praktisi **"Bridging Global Infrastructure to Address Local Challenges in HPC"**
(Informatics Practitioner Lecture Series, Informatika Universitas Bengkulu) oleh Dr. Warawan Diaz Carballo
(Thammasat University, Thailand).

## 1. Latar belakang & tantangan lokal
Pemateri menekankan peran **integrator**: orang yang menjembatani *masalah lokal* dengan *infrastruktur HPC*
lewat sebuah **workflow** (data masuk -> format -> transfer ke supercomputer -> model/skrip -> hasil -> aplikasi).

Tantangan lokal yang diangkat di sini: **penipuan berbasis manipulasi citra** (bukti transfer palsu, foto
produk/KTP hasil editan) yang marak di transaksi online, termasuk di UMKM. Memeriksa citra satu per satu
tidak skalabel; saat volume citra membesar (ribuan-jutaan), pemrosesan perlu **paralel** dan layak dijalankan
di klaster HPC memakai *batch script* (SLURM) seperti pada demo pemateri.

## 2. Pemetaan materi pemateri -> implementasi
| Materi pemateri | Implementasi di repo ini |
|---|---|
| Simulasi/komputasi skala besar butuh banyak operasi floating point | ELA berbasis operasi array float (NumPy) |
| Workflow integrator: data -> skrip -> job -> unduh hasil | `generate_dataset.py` -> `detect.py` -> `results/*.csv` |
| Batch script, minta resource (node/core), submit job | `slurm/job.sh` |
| Skalabilitas, performa, benchmarking (kompetisi HPC) | `benchmark.py` (speedup & efisiensi) |
| Artifact description/evaluation: dapat diulang & dipakai ulang | dataset sintetis ber-seed tetap, langkah run jelas |
| Kolaborasi lintas domain | modul deteksi dapat diganti pakar forensik / keamanan siber |

## 3. Metode
**Error Level Analysis (ELA)**: gambar disimpan ulang sebagai JPEG kualitas 75, lalu dihitung selisih piksel
terhadap aslinya. Area hasil tempelan dari sumber yang riwayat kompresinya berbeda memiliki *error level*
berbeda. Peta selisih dibagi blok 16x16; skor `frac_low` = porsi blok dengan error jauh di bawah median
(< 0,6 x median). Gambar ditandai mencurigakan bila `frac_low` > ambang (default 0,02).

**Paralelisasi**: tiap gambar independen (*embarrassingly parallel*) sehingga dibagi ke banyak proses dengan
`multiprocessing.Pool`. Kinerja diukur dengan speedup `S(p) = T(1)/T(p)` dan efisiensi `E(p) = S(p)/p`.

## 4. Cara menjalankan
```bash
pip install -r requirements.txt
python src/generate_dataset.py --out data --n 100 --size 512
python src/detect.py --data data --workers 4
python src/benchmark.py --data data --workers 1,2,4,8
```
Di klaster HPC: `sbatch slurm/job.sh` (sesuaikan partition/account).

## 5. Keluaran
- `results/detections.csv` : skor per gambar + label
- `results/benchmark.csv`, `results/speedup.png` : speedup & efisiensi

## 6. Catatan hasil & keterbatasan
- Pada dataset sintetis (100 asli + 100 manipulasi, seed 42) sebaran `frac_low` kedua kelas terpisah cukup baik
  dengan ambang 0,02. Angka akurasi/speedup **jalankan sendiri dan isi tabel di bawah** dengan hasil mesin Anda
  (speedup bergantung jumlah core; pada mesin 1 core tidak akan terlihat).
- Dataset sintetis sengaja disederhanakan. ELA bukan detektor universal: gagal bila gambar dikompres ulang
  berkali-kali, di-resize, atau diedit tanpa jejak kompresi berbeda. Langkah lanjut: dataset nyata (CASIA v2,
  Columbia), fitur tambahan (noise residual, copy-move), dan model CNN.

| Worker | Waktu (s) | Speedup | Efisiensi |
|---|---|---|---|
| 1 | | | |
| 4 | | | |
| 8 | | | |
