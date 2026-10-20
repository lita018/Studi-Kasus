# Studi Kasus: Deteksi Manipulasi Citra (PCD-Cyber) dengan Alur CRISP-DM

Studi kasus ini mengadaptasi materi kuliah praktisi **"Make Sense of Data with Analysis and AI"**
(Andria Arisal, BRIN, Informatics Practitioner Lecture Series, Prodi Informatika Universitas Bengkulu)
ke topik pengolahan citra digital untuk keamanan siber: **mendeteksi gambar yang dimanipulasi**
(contoh nyata: bukti transfer atau dokumen yang diedit).

> **Catatan:** dataset ini **sintetik** (dibuat oleh `generate_dataset.py`), seperti
> "PT Maju Bersama" di materi. Angka hasilnya menunjukkan cara kerja pipeline, **bukan** performa
> pada gambar dunia nyata.

## Pertanyaan bisnis → pertanyaan analitik

| | |
|---|---|
| Pertanyaan bisnis | "Gambar kiriman pelanggan ini bisa dipercaya atau tidak?" |
| Pertanyaan analitik | Dari fitur forensik citra, gambar mana yang berpotensi dimanipulasi dan harus diperiksa manual? |
| Unit analisis | 1 gambar |
| Target | `label` (0 = asli, 1 = dimanipulasi) |
| Masalah data | Kelas tidak seimbang (~15% palsu), sama seperti kasus invoice telat di demo (13%) |

## Pemetaan ke materi

| Konsep di materi | Implementasi di repo ini |
|---|---|
| CRISP-DM | Alur file di bawah |
| Garbage in, garbage out | Dataset dibuat dengan riwayat kompresi yang sama untuk kedua kelas (hindari leakage) |
| Data leakage | Tidak ada fitur yang "mengintip" label; ambang dipilih di data latih (CV), data uji hanya untuk evaluasi akhir |
| Data tidak seimbang & model malas | `baseline_mayoritas`: akurasi 85% tapi recall 0% |
| 4 level analitik | Descriptive, Diagnostic, Predictive, Prescriptive di `run_analysis.py` |
| Manusia memutuskan | Model hanya menyaring; gambar di atas ambang dikirim ke **review manual** |

## Cara menjalankan

```bash
pip install -r requirements.txt
python src/generate_dataset.py --n 1000      # buat gambar sintetik + metadata.csv
python src/run_analysis.py                   # ekstraksi fitur, model, evaluasi, grafik
```

Hasil tersimpan di `results/` (CSV, grafik PNG, `summary.json`). Seed tetap (42), jadi hasilnya sama setiap dijalankan.

## Struktur

```
src/generate_dataset.py   # data sintetik: copy_move, splice, text_edit
src/features.py           # ELA, residu noise, deteksi copy-move (12 fitur)
src/run_analysis.py       # pipeline analisis end-to-end
data/metadata.csv         # label & jenis manipulasi
results/                  # keluaran analisis
```

## Fitur forensik

- **ELA (Error Level Analysis):** area dengan riwayat kompresi berbeda memberi error rekompresi berbeda.
- **Residu noise:** area hasil tempel atau timpa biasanya punya pola noise yang tidak konsisten.
- **Copy-move:** pasangan blok bertekstur yang hampir identik tetapi berjauhan.

## Hasil (seed 42, 1000 gambar, data uji 300)

| Model (ambang 0.5) | Akurasi | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| Baseline mayoritas | 0.850 | 0.000 | 0.000 | 0.150 |
| Logistic Regression | 0.723 | 0.317 | 0.733 | 0.488 |
| Random Forest | 0.873 | 0.733 | 0.244 | 0.568 |

**Prescriptive:** dengan asumsi biaya gambar palsu lolos = 10 dan biaya review manual = 1,
ambang optimal dari validasi silang di data latih adalah **0,12**. Pada data uji:

| Kebijakan | Biaya total |
|---|---|
| Loloskan semua | 450 |
| Review manual semua | 300 |
| Random Forest @0,5 | 355 |
| Random Forest @0,12 | **182** |

Pada ambang 0,12, recall 95,6% (43 dari 45 gambar palsu tertangkap) dengan konsekuensi 119 gambar asli ikut ditandai (precision 26,5%).

**Diagnostic:** recall per jenis manipulasi pada ambang tsb: `text_edit` 100%, `splice` 100%, `copy_move` 83%.
Copy-move paling sulit karena sumber dan hasil salinannya berasal dari gambar yang sama.

## Pelajaran

1. Akurasi 85% bisa dicapai model yang tidak mendeteksi apa pun. Lihat **recall** dan **PR-AUC**.
2. Ambang batas adalah keputusan bisnis, bukan 0,5 otomatis. Biayanya menentukan titik optimal.
3. Model lebih cocok sebagai penyaring awal, keputusan akhir tetap oleh manusia.

## Keterbatasan

- Data sintetik dan kecil (128×128 piksel); fitur dibuat manual dan sederhana.
- Satu gambar per sumber, jadi tidak ada risiko sumber sama di data latih dan uji. Pada dataset nyata
  yang punya pasangan asli-palsu, **pisahkan data latih/uji per sumber** (`GroupShuffleSplit`).
- Asumsi biaya (10 : 1) hanya contoh.
