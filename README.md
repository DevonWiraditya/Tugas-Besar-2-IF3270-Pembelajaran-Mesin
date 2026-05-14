# Tugas Besar 2 IF3270 Pembelajaran Mesin

Repository ini berisi implementasi tugas besar Pembelajaran Mesin untuk eksperimen CNN pada dataset Intel Image Classification. Branch `feat/cnn` berisi pekerjaan CNN, sedangkan fondasi bersama ada di branch `dev`.

## Struktur Folder

```text
configs/
  cnn/                 Konfigurasi eksperimen CNN
  captioning/          Placeholder konfigurasi RNN/LSTM
dataset/               Dataset Intel Image Classification
notebooks/             Notebook analisis dan demo CNN
reports/               Hasil eksperimen, tabel analisis, dan visualisasi
scripts/               Entry point untuk menjalankan pipeline
src/
  cnn/                 Implementasi CNN Keras dan forward propagation NumPy
  captioning/          Placeholder bagian RNN/LSTM
  utils/               Utility bersama
models/                Artifact model hasil training lokal, tidak ikut GitHub
```

## Setup

Gunakan Python 3.10 atau versi yang kompatibel dengan TensorFlow yang dipakai di environment lokal.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dataset diharapkan tersedia dengan struktur berikut:

```text
dataset/
  seg_train/
  seg_test/
  seg_pred/
```

Validasi konfigurasi dan dataset:

```powershell
python scripts\validate_cnn_config.py --check-data
python scripts\inspect_cnn_dataset.py --sample-batch
```

## Pipeline CNN

Melatih 16 variasi CNN shared parameter:

```powershell
python scripts\train_cnn.py --parameter-sharing shared
```

Menganalisis hasil 16 eksperimen shared:

```powershell
python scripts\analyze_cnn_results.py
```

Melatih arsitektur non-shared dari arsitektur shared terbaik:

```powershell
python scripts\train_cnn.py --parameter-sharing nonshared --best-from-summary --early-stopping-patience 3
```

Membandingkan shared dan non-shared parameter:

```powershell
python scripts\compare_cnn_parameter_sharing.py --require-nonshared
```

Membandingkan forward propagation Keras dan from scratch NumPy:

```powershell
python scripts\compare_cnn_scratch.py --batch-size 32 --output-path reports\cnn\shared_scratch_test_comparison.csv
```

Mengumpulkan history training untuk analisis notebook:

```powershell
python scripts\collect_cnn_histories.py
```

Membuat visualisasi feature map dan Grad-CAM:

```powershell
python scripts\generate_cnn_visualizations.py
```

## Notebook

Notebook utama untuk demo dan analisis CNN ada di:

```text
notebooks/cnn_analysis.ipynb
```

Notebook tersebut menampilkan dataset, 16 eksperimen shared, pengaruh hyperparameter, perbandingan shared vs non-shared, perbandingan Keras vs from scratch, serta visualisasi feature map dan Grad-CAM.

## Catatan Artifact Model

Folder `models/` tidak ikut dipush ke GitHub karena ukurannya besar dan berisi bobot hasil training. Untuk menjalankan ulang evaluasi yang membutuhkan bobot model, jalankan pipeline training terlebih dahulu atau gunakan artifact model yang dibagikan terpisah melalui cloud storage.

Hasil ringkasan eksperimen yang dibutuhkan untuk analisis sudah tersedia di folder `reports/`, sedangkan demo dan pembahasan utama ada di `notebooks/cnn_analysis.ipynb`.
