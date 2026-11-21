# Menjalankan E7_T

Folder publik memuat kode dan hasil tersimpan. Dataset, tensor gambar, mask mentah dan semua checkpoint disiapkan sendiri di lingkungan lokal. README utama serta E7B15_Final.ipynb dapat dibaca tanpa bobot model atau GPU.

## Instalasi

Gunakan lingkungan Python yang sesuai dengan requirements.txt. Paket E7 asli mencatat Python 3.11, PyTorch 2.8.0 dan CUDA 12.6. Untuk mereproduksi lingkungan itu pada mesin NVIDIA yang kompatibel:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Di PowerShell, aktivasi environment menggunakan .venv\Scripts\Activate.ps1. requirements.txt berisi dependensi utama E7_T. Dependensi khusus segmentasi dan pretrained backbone dijelaskan pada bagian instalasi notebook ablation masing-masing.

## Bangun ulang cache P3

Siapkan chest_xray dengan folder train, val, test dan dua kelas NORMAL serta PNEUMONIA. Split eksperimen berjumlah 4.646 / 586 / 624. Siapkan cache mask U-Net dari eksperimen segmentasi, termasuk provenance-nya.

```bash
python preprocess_p3.py --dataset /path/to/chest_xray --mask-dir /path/to/masks/unet_resnet34 --output p3_cache
```

Alternatifnya, gunakan bobot segmentasi resmi melalui --unet-weights. Paket opsional yang dipakai sumber adalah lungs-segmentation==1.1.2. Notebook menjelaskan pemasangan --no-deps karena metadata dependensi paket tersebut memakai PyTorch versi lama.

```bash
python -m pip install --no-deps lungs-segmentation==1.1.2
python preprocess_p3.py --dataset /path/to/chest_xray --unet-weights /path/to/resnet34.pth --device cuda:0 --output p3_cache
```

[Sumber bobot segmentasi resmi](https://github.com/alimbekovKZ/lungs_segmentation/releases/tag/1.0.0).

Script membentuk CLAHE + hard mask + letterbox 224 × 224 + augmentasi statis khusus train. P3 mempertahankan skala 0–255. Hash hasil dibandingkan dengan data/p3_cache_manifest.json. Perbedaan split, urutan sumber, versi library atau inferensi mask dapat membuat verifikasi cache gagal. data/source_index.csv dan data/static_augmentation_plan.csv menyimpan urutan serta rencana transformasi arsip, tanpa tensor gambar.

## Training baru

Jalankan dari root repository setelah cache lokal siap:

```bash
python train_pneunet_e7_t.py --cache p3_cache --smoke
python train_pneunet_e7_t.py --cache p3_cache --epochs 50 --output runs/seed42
```

Perintah tersebut hanya petunjuk. Penyiapan repository ini tidak menjalankannya. Trainer memakai Weighted BCE + L2 eksplisit, Adam, batch fisik 32, FP32, seed 42, maksimum 50 epoch dan patience 7. Epoch dipilih berdasarkan validation macro-F1. Early stopping mengikuti validation loss. Direktori run yang sudah ada ditolak agar hasil tidak tertimpa.

```bash
python train_pneunet_e7_t.py --resume runs/seed42 --epochs 50
python train_pneunet_e7_t.py --cache p3_cache --evaluate /path/to/best.pt --history outputs/history.csv --output runs/re_evaluation
python benchmark_e7_t_latency.py --checkpoint runs/seed42/best.pt --outputs runs/seed42/outputs --cache p3_cache
```

Semua checkpoint yang dihasilkan tersimpan lokal dan diabaikan Git. Bobot arsip E7_T tidak disertakan di repository publik. Metadata serta checksum aslinya tetap ada di outputs/selection.json dan outputs/archived_run_result.json.

## Melihat hasil

Buka E7B15_Final.ipynb untuk membaca outputs/ yang sudah disertakan. Penampil itu tidak membutuhkan dataset, checkpoint atau GPU. Untuk run baru, ubah RESULTS_DIR pada notebook menjadi runs/seed42/outputs.

## Notebook ablasi

Keempat notebook di ablation/ disalin tanpa menghapus output historis atau mengganti konfigurasi riset. Sebagian konfigurasi tersimpan masih mengaktifkan training atau inferensi. Baca hasil tersimpan tanpa Run All. Untuk eksekusi baru, atur WORKSPACE, dataset, mask, sumber notebook dan checkpoint pada konfigurasi notebook, lalu pilih mode yang diinginkan. Jalur historis server dipertahankan sebagai bagian dari arsip dan tidak otomatis menunjuk ke checkout baru.

benchmark_P3_seed42.ipynb membaca definisi dari notebook segmentasi dan memerlukan checkpoint E5 P3 lokal. Ketiga model pembanding memakai bobot pretrained eksternal. File output yang menyebut best.pt mendeskripsikan artefak historis, bukan file yang ikut dibagikan.
