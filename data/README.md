# Metadata data

Folder ini berisi informasi kecil untuk menelusuri urutan sumber, konfigurasi preprocessing dan hasil pembanding. Tidak ada citra dataset atau tensor piksel di folder ini.

| File | Kegunaan |
|---|---|
| p3_cache_manifest.json | Checksum cache P3 arsip, provenance mask dan konfigurasi acuan |
| source_index.csv | Identitas sumber, label, split dan urutan contoh yang membentuk tensor |
| static_augmentation_plan.csv | Penetapan augmentasi statis beserta sudut atau sigma |
| baseline_classification_metrics.csv | Metrik referensi E5 dan backbone pembanding pada P3 |

Path server yang tercatat dalam metadata adalah jejak sumber eksperimen. Lokasi tersebut tidak otomatis tersedia pada komputer pembaca. Tensor train/val/test.npy dan mask gambar tidak disertakan. [Petunjuk membentuk cache lokal](../docs/REPRODUCTION.md#bangun-ulang-cache-p3).
