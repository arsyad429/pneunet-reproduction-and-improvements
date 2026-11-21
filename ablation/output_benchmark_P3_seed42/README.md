# Benchmark model pada P3

PneuNet E5 P3 seed 42 dibandingkan dengan DenseNet121-RSNA, RAD-DINO dan DINOv3 ViT-L/16 pada split yang sama. Notebook ini juga memuat evaluasi, atribusi, VRAM dan efisiensi GPU.

[Notebook sumber](../benchmark_P3_seed42.ipynb) · [Peta seluruh studi](../README.md)

Folder ini memuat artefak dari output_benchmark_P3_seed42/output_benchmark_P3_seed42. Susunan internal hasil dipertahankan, sementara pembungkus folder unduhan yang berulang sudah dirapikan. Semua CSV, JSON, PNG dan NPZ heatmap yang tersedia disertakan. Bobot checkpoint dan duplikat otomatis notebook dikecualikan. Rujukan best.pt pada metadata merupakan lokasi eksperimen asli dan tidak menyatakan bahwa bobot ikut disertakan.

README pada setiap subfolder menjelaskan tipe hasilnya. File konfigurasi dan result/summary menjadi sumber angka penelitian. Gambar Grad-CAM merupakan analisis kontribusi fitur, bukan mask anotasi ahli atau pengukuran akurasi segmentasi.
