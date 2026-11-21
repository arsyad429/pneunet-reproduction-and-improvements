# Ablasi preprocessing E5

Lima kondisi dengan arsitektur E5 dan batch fisik 32: RAW 0–255, normalisasi /255, CLAHE, rotasi 50% train, dan Gaussian blur 50% train. Rotasi dan blur tidak diterapkan pada validation/test utama.

[Notebook sumber](../reproduce_pneunet_e5_data_ablation_batch32.ipynb) · [Peta seluruh studi](../README.md)

Folder ini memuat artefak dari output_pneunet_e5_data_ablation_batch32. Susunan internal hasil dipertahankan, sementara pembungkus folder unduhan yang berulang sudah dirapikan. Semua CSV, JSON, PNG dan NPZ heatmap yang tersedia disertakan. Bobot checkpoint dan duplikat otomatis notebook dikecualikan. Rujukan best.pt pada metadata merupakan lokasi eksperimen asli dan tidak menyatakan bahwa bobot ikut disertakan.

README pada setiap subfolder menjelaskan tipe hasilnya. File konfigurasi dan result/summary menjadi sumber angka penelitian. Gambar Grad-CAM merupakan analisis kontribusi fitur, bukan mask anotasi ahli atau pengukuran akurasi segmentasi.
