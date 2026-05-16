# Tugas Besar 2 IF3270 Pembelajaran Mesin

## Deskripsi Singkat

Repository ini berisi implementasi dua jalur utama untuk Tugas Besar 2 IF3270 Pembelajaran Mesin:
- CNN untuk image classification pada dataset Intel Image Classification
- Simple RNN dan LSTM untuk image captioning pada dataset Flickr8k

Kode utama berada di dalam folder `src/`. Runner script berada di `src/scripts/`, sedangkan notebook analisis berada di `src/notebooks/`.

## Setup

1. Buat dan aktifkan virtual environment
2. Install dependency

```bash
pip install -r requirements.txt
```

Jika menggunakan environment GPU terpisah, pastikan TensorFlow dan dependency lain sudah terpasang di environment tersebut sebelum menjalankan script.

## Struktur Singkat

- `src/cnn/` : implementasi CNN
- `src/captioning/` : implementasi captioning
- `src/scripts/` : runner script
- `src/notebooks/` : notebook analisis
- `configs/` : file konfigurasi
- `artifacts/` : output preprocessing dan feature extraction lokal
- `models/` : model hasil training lokal
- `reports/` : summary, ranking, dan output evaluasi

## Catatan Model

Folder `models/` tidak dipush ke GitHub karena berisi bobot hasil training berukuran besar. Beberapa file model melebihi batas ukuran file GitHub biasa, sehingga memasukkannya ke repository dapat membuat proses clone, pull, dan push menjadi berat atau gagal.

## Cara Menjalankan CNN

Bagian CNN digunakan untuk image classification pada dataset Intel Image Classification. Pipeline CNN mencakup pelatihan 16 variasi model Conv2D shared parameter, analisis hyperparameter, perbandingan shared vs non-shared parameter, perbandingan Keras vs forward propagation from scratch, serta visualisasi feature maps dan Grad-CAM.

Validasi konfigurasi:

```bash
python src/scripts/validate_cnn_config.py --check-data
```

Melihat ringkasan dataset:

```bash
python src/scripts/inspect_cnn_dataset.py --sample-batch
```

Melatih 16 variasi CNN shared parameter:

```bash
python src/scripts/train_cnn.py --parameter-sharing shared
```

Menganalisis hasil eksperimen shared:

```bash
python src/scripts/analyze_cnn_results.py
```

Melatih model non-shared dari arsitektur shared terbaik:

```bash
python src/scripts/train_cnn.py --parameter-sharing nonshared --best-from-summary --early-stopping-patience 3
```

Membandingkan shared dan non-shared parameter:

```bash
python src/scripts/compare_cnn_parameter_sharing.py --require-nonshared
```

Membandingkan Keras dengan forward propagation from scratch:

```bash
python src/scripts/compare_cnn_scratch.py --batch-size 32 --output-path reports/cnn/shared_scratch_test_comparison.csv
```

Mengumpulkan history training untuk notebook:

```bash
python src/scripts/collect_cnn_histories.py
```

Membuat visualisasi feature maps dan Grad-CAM:

```bash
python src/scripts/generate_cnn_visualizations.py
```

Hasil analisis CNN yang sudah dihasilkan tersimpan di `reports/cnn/`, sedangkan notebook pembahasan berada di `src/notebooks/cnn_analysis.ipynb`. Folder `models/` tidak dipush ke GitHub karena berisi bobot model berukuran besar; jika diperlukan, model perlu dilatih ulang atau dibagikan sebagai artifact terpisah.

## Cara Menjalankan Captioning

Preprocessing caption:

```bash
python src/scripts/prepare_flickr8k.py
```

Feature extraction:

```bash
python src/scripts/extract_caption_features.py --split train
python src/scripts/extract_caption_features.py --split validation
python src/scripts/extract_caption_features.py --split test
```

Training captioning:

```bash
python src/scripts/train_captioning.py
```

Evaluasi captioning:

```bash
python src/scripts/evaluate_captioning.py --run-id rnn_preinject_layers2_hidden128 --split test
python src/scripts/evaluate_captioning.py --run-id lstm_preinject_layers2_hidden128 --split test
```

Analisis hasil captioning:

```bash
python src/scripts/analyze_captioning_results.py
```

Eksperimen panjang maksimum caption:

```bash
python src/scripts/run_caption_length_experiments.py --run-id rnn_preinject_layers2_hidden128 --decoder-type rnn
python src/scripts/run_caption_length_experiments.py --run-id lstm_preinject_layers2_hidden128 --decoder-type lstm
```

## Notebook

- `src/notebooks/cnn_analysis.ipynb`
- `src/notebooks/captioning_analysis.ipynb`

Untuk notebook captioning, gunakan kernel environment yang sesuai dengan hasil eksperimen captioning yang sudah dijalankan.

## Pembagian Tugas

| Nama | Tugas |
|---|---|
| Muhammad Adam Mirza | Mengerjakan seluruh bagian image captioning RNN/LSTM, termasuk preprocessing, training, scratch implementation, evaluasi, dan analisis. |
| Devon Wiraditya T. | Mengerjakan seluruh bagian CNN, termasuk preprocessing, training, scratch implementation, evaluasi, analisis, serta membantu finalisasi repository dan laporan. |
