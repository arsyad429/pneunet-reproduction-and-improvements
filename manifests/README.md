# Inventaris paket

- [artifact_inventory.csv](artifact_inventory.csv) mencatat file yang disalin, sumber relatif workspace, lokasi dalam repository, ukuran dan SHA-256. Notebook, kode E7 dan hasil disalin tanpa eksekusi ulang.
- [excluded_artifacts.csv](excluded_artifacts.csv) mencatat bobot model, cache tensor gambar dan duplikat otomatis yang dikecualikan.
- [preparation_report.json](preparation_report.json) merangkum jumlah artefak, ukuran, lokasi paket asal di luar repo dan hasil verifikasi penataan.

README dan dokumen penataan yang dihasilkan baru tidak masuk inventaris salinan. Teks arsitektur pada docs/ berasal dari sumber dengan tautan lokal yang diperbaiki. Checksum dalam JSON hasil eksperimen tetap dipertahankan dan tidak diganti menjadi checksum salinan baru.

## Publikasi GitHub

[publication_inventory.csv](publication_inventory.csv) memuat checksum versi file saat publikasi. Inventaris ini mengecualikan dirinya sendiri. artifact_inventory.csv tetap menjadi snapshot salinan pada tahap persiapan awal. Notebook penampil E7B15_Final.ipynb telah berubah setelah tahap persiapan dan versi yang ada sekarang dipertahankan.
