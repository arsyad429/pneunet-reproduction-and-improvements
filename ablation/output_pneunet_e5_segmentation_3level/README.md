# Segmentasi paru-paru

Dua segmenter, tiga perlakuan ROI, P1–P3 dan kontrol tanpa segmentasi. Tahap awal memiliki 21 konfigurasi seed 42. Bagian lanjutan menguji hard mask U-Net pada P1–P4 dengan seed 42, 123, dan 3407.

[Notebook sumber](../PneuNet_E5_3Level_Segmentation_Ablation_FIXED.ipynb) · [Peta seluruh studi](../README.md)

Folder ini memuat artefak dari output_pneunet_e5_segmentation_3level (1)/output_pneunet_e5_segmentation_3level. Susunan internal hasil dipertahankan, sementara pembungkus folder unduhan yang berulang sudah dirapikan. Semua CSV, JSON, PNG dan NPZ heatmap yang tersedia disertakan. Bobot checkpoint dan duplikat otomatis notebook dikecualikan. Rujukan best.pt pada metadata merupakan lokasi eksperimen asli dan tidak menyatakan bahwa bobot ikut disertakan.

README pada setiap subfolder menjelaskan tipe hasilnya. File konfigurasi dan result/summary menjadi sumber angka penelitian. Gambar Grad-CAM merupakan analisis kontribusi fitur, bukan mask anotasi ahli atau pengukuran akurasi segmentasi.
