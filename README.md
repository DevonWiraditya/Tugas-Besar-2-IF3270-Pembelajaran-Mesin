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
- `artifacts/` : output preprocessing dan feature extraction
- `models/` : model hasil training
- `reports/` : summary, ranking, dan output evaluasi

## Cara Menjalankan CNN

Validasi konfigurasi:

```bash
python src/scripts/validate_cnn_config.py --check-data
```

Training CNN:

```bash
python src/scripts/train_cnn.py
```

tambahin wet sesuai ama pny lu

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

### Anggota 1
...

### Anggota 2
...
