# PneuNet E7-B + GRN (T)

## Tujuan eksperimen dan keputusan

Eksperimen mengembangkan classifier ringan dengan masukan P3 yang sama dengan PneuNet E5. Sasaran eksploratifnya adalah Macro-F1 test di atas 0,950122 dengan maksimum 1,5 juta parameter. E5 memiliki 785.192 parameter. Dua blok residual context E5 menyumbang sekitar 590.848 parameter dan bekerja pada feature map 112×112. Mengganti blok ini memberi ruang penghematan paling besar sambil mempertahankan ekstraksi fitur awal dan penggabungan multiskala.

1. **E5 ke E7-B.** Masing-masing dari dua blok context diganti bottleneck `1×1: 128→64`, `depthwise 3×3: 64→64`, dan `1×1: 64→128`. Konvolusi 1×1 mencampur kanal, sedangkan konvolusi depthwise memproses tiap kanal secara spasial dengan biaya yang lebih rendah. Dilasi blok pertama 1 dan blok kedua 2 memperluas jangkauan spasial tanpa mengubah grid 112×112. Shortcut residual, BatchNorm, dan ReLU tetap mengikuti implementasi E7-B. Kedua blok memakai 34.944 parameter. Total classifier turun menjadi 229.288 parameter. Pemisahan konvolusi spasial dan pencampuran kanal mengikuti gagasan depthwise separable convolution. [MobileNets](https://arxiv.org/abs/1704.04861)
2. **E7-B ke T.** Eksperimen sebelumnya memvariasikan lebar context, GroupNorm, stride, dan spatial gating. Kandidat J mencapai 0,948975 dengan 1.356.456 parameter. Pemadatan feature map ke 56×56 pada varian J/Z menurunkan Macro-F1 test pada percobaan seed 42. Percobaan spatial gating pada N juga menurunkan Macro-F1 internal sekitar 0,06464 pada anggaran screening yang sama. Hasil tersebut memotivasi uji perubahan kecil pada E7-B asli sambil mempertahankan resolusi fitur dan BatchNorm. T menambahkan Global Response Normalization (GRN) setelah context dan sebelum pooling untuk menguji apakah respons kanal per citra dapat membantu membedakan struktur normal dari fitur yang mendukung prediksi pneumonia. [Riwayat percobaan](../outputs/all_search_history.csv)
3. **Alasan penempatan dan inisialisasi.** GRN bekerja pada fitur yang sudah merangkum cabang ASPP dan dua blok context. Modul mengubah bobot relatif fitur sebelum peta perhatian dan pooling dihitung, sehingga perubahan bisa memengaruhi representasi yang diterima classifier. Dua vektor GRN diinisialisasi nol. Output awal sama dengan E7-B dan seluruh bobot awal backbone dipertahankan pada seed yang sama. Uji sebelum training mencocokkan tensor backbone serta logit awal secara eksak, dan memeriksa gradien finite pada input biasa dan energi nol. GRN menambah 256 parameter, sehingga total T menjadi 229.544 parameter, sekitar 70,8% lebih sedikit daripada E5. Mekanisme GRN berasal dari ConvNeXt V2 dan diterapkan di sini sebagai modifikasi terkontrol dengan layout NCHW. [ConvNeXt V2](https://arxiv.org/abs/2301.00808) · [Implementasi T](../pneunet_e7_t.py)

## Pipeline P3

1. Citra X-ray dibaca sebagai grayscale. Mask paru biner U-Net ResNet34 diambil dari cache E5 P3 pada geometri asli dengan threshold 0,2.
2. CLAHE memakai clip limit 2,5 dan tile grid 8×8. Hard masking mengisi area di luar mask dengan nol. Varian `unet_resnet34__mask` mempertahankan geometri citra sebelum letterbox.
3. Letterbox mengubah ukuran secara isotropik dan memusatkan hasil pada kanvas 224×224 berpadding nol.
4. Train mendapat penugasan augmentasi statis seed 42 per kelas dengan proporsi sekitar 50% tanpa transformasi, 25% rotasi ±15°, dan 25% Gaussian blur dengan sigma 0,5 hingga 1,5. Validation dan test memakai citra tanpa augmentasi.
5. Input T dan E5 berupa tensor grayscale FP32 `[B, 1, 224, 224]` pada rentang 0–255. Replikasi menjadi tiga kanal dilakukan di dalam model. Normalisasi khusus baseline DenseNet, RAD-DINO, dan DINOv3 dijalankan oleh wrapper masing-masing setelah menerima masukan P3.

Cache lokal menyimpan hasil akhir tahap 1–4 sebagai tensor piksel `uint8` siap dipakai, dengan 4.646 train, 586 validation, dan 624 test. Loader mengonversi ke FP32 dan mempertahankan piksel, urutan indeks, serta label `NORMAL=0`, `PNEUMONIA=1`. Setiap piksel dari 5.856 contoh telah dicocokkan saat ekspor dengan loader P3 penelitian. Augmentasi termasuk dalam cache train. Metadata mencatat asal dan checksum mask, cache asli, rencana augmentasi, konfigurasi, dan checkpoint E5. [Manifest P3](../data/p3_cache_manifest.json) · [Rencana augmentasi](../data/static_augmentation_plan.csv)

## Arsitektur T dan aliran tensor

| Tahap | Output tanpa dimensi batch | Operasi |
|---|---:|---|
| Input dan stem | 32 × 224 × 224 | Replikasi grayscale dan Conv 3×3, BatchNorm, ReLU |
| Detail refinement | 32 × 224 × 224 | Dua blok residual detail |
| Cabang ekstraksi dan concat | 128 × 224 × 224 | Satu cabang 1×1 dan tiga cabang depthwise separable dengan SE |
| Downsample | 128 × 112 × 112 | Depthwise stride 2 dan dropout |
| ASPP dan fusi | 128 × 112 × 112 | Empat cabang 32 kanal, yakni 1×1 dan 3×3 dilasi 1, 3, 6, lalu fusi 1×1 |
| Context refinement | 128 × 112 × 112 | Dua bottleneck residual 128→64→128, dilasi spasial 1 dan 2 |
| GRN | 128 × 112 × 112 | Modulasi kanal dari respons energi relatif per gambar |
| Pooling dan head | 128 → 1 | Normalized learnable spatial pooling, dropout 0,3, dan linear untuk logit PNEUMONIA |

ASPP memakai konvolusi atrous untuk menggabungkan informasi dari beberapa jangkauan spasial. T mempertahankan konfigurasi ASPP E5. [DeepLabv3](https://arxiv.org/abs/1706.05587) · [Inventaris layer](../outputs/layer_inventory.csv)

Pada setiap blok context, jalur transformasinya adalah `Conv1×1 → BN → ReLU → DWConv3×3(d) → BN → ReLU → Conv1×1 → BN`. Output blok adalah `ReLU(x + transform(x))`. Konvolusi depthwise memakai `groups=64` dan padding sebesar dilasi, sehingga dimensi H×W tetap.

Untuk fitur context `X[b,c,h,w]`, GRN menghitung `e[b,c] = sqrt(sum_hw(X²))`, lalu `r[b,c] = e[b,c] / (mean_c(e[b,c]) + 1e-6)`. Outputnya `Y = X + γ[c] × X × r[b,c] + β[c]`. `γ` dan `β` merupakan parameter yang dipelajari per kanal. Energi dihitung per contoh dan membandingkan respons kanal relatif terhadap kanal lain. Karena `γ=β=0` saat inisialisasi, `Y=X` pada awal training.

Pooling menghitung peta `a = sigmoid(Conv1×1(ReLU(Conv1×1(Y))))`. Vektor 128 kanal adalah `sum_hw(a × Y) / max(sum_hw(a), 1e-6)`. Head menghasilkan satu logit. Probabilitas PNEUMONIA adalah sigmoid dari logit dan dipetakan ke kelas 1 pada threshold 0,5.

## Training, evaluasi, dan interpretasi

Training T dimulai dari inisialisasi acak dengan seed 42. Konfigurasi menggunakan physical batch 32, FP32, Adam dengan learning rate 0,001 dan weighted BCE dengan bobot kelas inverse-frequency. L2 sebesar 0,001 diterapkan pada kernel Conv/Linear menurut konvensi eksperimen. Early stopping memakai validation loss dengan patience 7. Checkpoint dipilih berdasarkan Macro-F1 validation tertinggi. T selesai pada epoch 15 dengan checkpoint terpilih epoch 6 dan Macro-F1 validation 0,911747. [Konfigurasi run](../outputs/config.json) · [Riwayat epoch](../outputs/history.csv)

| Model | Parameter | Macro-F1 test | FP NORMAL | FN PNEUMONIA |
|---|---:|---:|---:|---:|
| T, E7-B + GRN | 229.544 | 0,950635 | 12 | 17 |
| E7-B awal | 229.288 | 0,946388 | 22 | 9 |
| E5 P3 | 785.192 | 0,934095 | 27 | 11 |
| DenseNet121-RSNA | 6.948.609 | 0,950122 | 18 | 11 |
| RAD-DINO | 86.581.249 | 0,925636 | 15 | 29 |
| DINOv3 ViT-L/16 | 303.130.625 | 0,947401 | 11 | 20 |

T menghasilkan `[[222,12],[17,373]]`, recall NORMAL 0,948718, recall PNEUMONIA 0,956410, dan AUROC 0,987552. Dibanding E5, FP turun 15 sementara FN naik 6. Dibanding E7-B awal, FP turun 10 sementara FN naik 8. Ini menunjukkan trade-off pada threshold yang dipertahankan. Aritmetika yang dihitung untuk Conv/Linear turun dari 22,81 GFLOPs E5 menjadi 8,87 GFLOPs T pada batch 1. Angka ini memakai 2 FLOP per MAC dan tidak mencakup seluruh pekerjaan normalisasi, pooling, serta operasi elemen GRN. Latency aktual diukur terpisah. [Prediksi test](../outputs/test_predictions.csv) · [Perbandingan benchmark](../outputs/final_comparison.csv)

Pemilihan epoch terjadi pada validation. Pemilihan kandidat yang memenuhi sasaran memakai hasil test seed 42 sesuai protokol eksplorasi yang diminta. Selisih terhadap DenseNet sekitar 0,0513 poin persentase pada 624 kasus. Generalisasi keuntungan GRN dan reliabilitas lintas seed perlu diuji pada kohort independen dan beberapa seed. Gambar Grad-CAM disediakan untuk memeriksa kontribusi fitur spasial dan tidak diperlakukan sebagai anotasi lesi. Peta asli dan versi dengan overlay mask tersedia untuk kedua kelas. [Grad-CAM PNEUMONIA](../outputs/gradcam_target_1_original.png) · [Ringkasan test](../outputs/test_metrics.json)

## Latency inferensi pada A100

CUDA Events mengukur forward FP32 pada input P3 test yang sudah ada di GPU. Waktu baca, segmentasi, preprocessing, pemuatan checkpoint, dan transfer input berada di luar pengukuran. Normalisasi di dalam wrapper baseline termasuk. Tiap ukuran batch memakai tiga sesi, masing-masing 30 warmup dan 100 pengukuran. AMP dan TF32 dimatikan.

| Model | Median batch 1 (ms/batch) | Median batch 32 (ms/batch) |
|---|---:|---:|
| DINOv3_ViTL16 | 181.58 | 441.96 |
| DenseNet121_RSNA | 96.74 | 99.19 |
| E7 T | 11.11 | 88.11 |
| PneuNet_E5_P3 | 12.09 | 102.46 |
| RAD_DINO | 85.68 | 154.42 |

Pengukuran di NVIDIA A100-SXM4-80GB. Aktivitas GPU lain tercatat pada 368 dari 390 sampel. Variasi antar sesi menjadi bagian dari hasil. Pengukuran GPU senggang diperlukan untuk menilai perbedaan kecepatan antararsitektur dengan gangguan yang lebih kecil. [Ringkasan CSV](../outputs/latency/20261120T023942_313817Z_y7u3yg4f/latency_summary.csv) · [Metadata](../outputs/latency/20261120T023942_313817Z_y7u3yg4f/environment.json)

## Reproduksi hasil

Pipeline P3 dapat dibangun ulang dari X-ray asli dengan mask arsip atau checkpoint U-Net ResNet34 resmi. Inferensi mask pada CUDA mengizinkan TF32 untuk konvolusi cuDNN agar sesuai dengan mask arsip. Pengaturan tersebut khusus untuk segmentasi sebelum training classifier. Script memeriksa urutan sumber, provenance mask, dan SHA-256 dari ketiga array sebelum menyimpan cache. [Script preprocessing](../preprocess_p3.py) · [Perintah reproduksi](REPRODUCTION.md#bangun-ulang-cache-p3)

Trainer membaca tensor P3 yang sudah diproses dan mencocokkan checksum dengan manifest versi repo. Setiap run menyimpan hasil aktual dari checkpoint terpilih dalam subfolder outputs. Perbandingan klasifikasi memakai baseline arsip pada data P3 yang sama. Grad-CAM run baru dihitung pada citra P3 dengan layer context_refine.response_norm. Pengukuran latency untuk checkpoint T baru dijalankan terpisah melalui script CUDA.
