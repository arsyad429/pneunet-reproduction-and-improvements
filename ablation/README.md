# Ablation

Folder ini mengumpulkan empat notebook riset dan hasil eksekusinya. Notebook utama E7_T serta hasil akhirnya ada di root repository dan outputs/.

| Notebook / studi | Isi | Folder hasil |
|---|---|---|
| [Segmentasi paru-paru](PneuNet_E5_3Level_Segmentation_Ablation_FIXED.ipynb) | Dua segmenter, tiga perlakuan ROI, P1–P3 dan kontrol tanpa segmentasi. Tahap awal memiliki 21 konfigurasi seed 42. Bagian lanjutan menguji hard mask U-Net pada P1–P4 dengan seed 42, 123, dan 3407. | [Output](output_pneunet_e5_segmentation_3level/) |
| [Faktorial CLAHE × augmentasi](PneuNet_E5_Factorial_2x2_CLAHE_Augmentation.ipynb) | Empat kondisi E5: baseline, CLAHE saja, augmentasi saja, serta CLAHE + augmentasi. Seluruh kondisi memakai normalisasi /255 agar skala intensitas sama. | [Output](output_pneunet_e5_factorial_2x2/) |
| [Ablasi preprocessing E5](reproduce_pneunet_e5_data_ablation_batch32.ipynb) | Lima kondisi dengan arsitektur E5 dan batch fisik 32: RAW 0–255, normalisasi /255, CLAHE, rotasi 50% train, dan Gaussian blur 50% train. Rotasi dan blur tidak diterapkan pada validation/test utama. | [Output](output_pneunet_e5_data_ablation_batch32/) |
| [Benchmark model pada P3](benchmark_P3_seed42.ipynb) | PneuNet E5 P3 seed 42 dibandingkan dengan DenseNet121-RSNA, RAD-DINO dan DINOv3 ViT-L/16 pada split yang sama. Notebook ini juga memuat evaluasi, atribusi, VRAM dan efisiensi GPU. | [Output](output_benchmark_P3_seed42/) |

Notebook dipertahankan sama persis dengan file sumber, termasuk kode dan output selnya. Hasil terpisah disalin utuh kecuali bobot model, cache citra, metadata OS dan salinan otomatis .ipynb_checkpoints. Snapshot eksekusi di folder hasil juga dipertahankan bila tersedia.

Untuk melihat hasil, buka notebook dan baca output yang sudah tersimpan. Konfigurasi beberapa notebook masih mengaktifkan training. Untuk run baru, sesuaikan jalur dataset, cache, sumber notebook, checkpoint dan dependensi sesuai petunjuk notebook. Dataset serta semua bobot harus disiapkan sendiri. [Panduan eksekusi](../docs/REPRODUCTION.md#notebook-ablasi).

Skala input berbeda antarstudi. Faktorial 2×2 selalu memakai /255. Ablasi lima kondisi memakai RAW 0–255 kecuali normalisasi. Studi segmentasi memakai /255 pada P1/P4 dan 0–255 pada P2/P3. Benchmark P3 membandingkan input dasar yang sama, lalu menerapkan normalisasi khusus backbone masing-masing.

Folder smoke dan audit berisi pemeriksaan teknis. Angkanya tidak menjadi hasil full training. Gunakan metadata run, konfigurasi dan nama split untuk menentukan hasil yang dibandingkan.
