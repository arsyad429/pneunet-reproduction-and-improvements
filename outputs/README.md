# Hasil PneuNet E7_T

Hasil kandidat T, seed 42. Semua file di sini adalah hasil yang telah tersedia sebelum penataan repository, bukan hasil training ulang.

| Artefak | Isi |
|---|---|
| selection.json, archived_run_result.json, config.json | Pemilihan kandidat/epoch, metadata run dan konfigurasi |
| history.csv, learning_curve.png | Loss dan metrik per epoch |
| train/val/test_metrics.json dan *_report.json | Metrik agregat serta classification report tiap split |
| *_predictions.csv | Prediksi, label dan probabilitas per citra |
| *_confusion_matrix.csv/png | Confusion matrix dalam angka dan gambar |
| final_comparison.csv | Perbandingan E7_T dengan hasil baseline P3 |
| gradcam_target_*.png | Grad-CAM asli dan tampilan dengan mask paru, dua target kelas |
| gradcam_original_heatmaps.npz | Array heatmap dan indeks contoh, tanpa bobot model |
| test_cases.csv, gradcam_lung_attribution.csv | Identitas contoh dan rasio atribusi pada area paru |
| all_search_history.csv, post_training_test_summary.csv | Riwayat pencarian kandidat dan evaluasinya |
| input_audit.json, dummy_smoke.json, layer_inventory.csv | Audit input, pemeriksaan teknis arsip dan inventaris arsitektur |
| latency/ | Benchmark forward GPU beserta percobaan dan metadata lingkungan |

Metadata asli mencatat test_used_for_selection=true untuk pemilihan kandidat. Interpretasi utama dan keterbatasannya ditulis di [README induk](../README.md#cara-membaca-hasil-penelitian). Dataset dan checkpoint model tidak ada di folder publik ini.
