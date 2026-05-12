# Tugas Besar 2 IF3270 Pembelajaran Mesin

## Tujuan README Ini

README ini ditulis sebagai **handoff context** untuk AI lain atau anggota tim lain yang akan melanjutkan repository ini.

Fokus README ini:

- menjelaskan requirement wajib dari spesifikasi,
- memisahkan requirement wajib dan bonus,
- memetakan kondisi repo saat ini berdasarkan isi kode yang benar-benar sudah ada,
- menjelaskan apa yang sudah selesai,
- menjelaskan apa yang belum selesai,
- memberi urutan kerja selanjutnya,
- dan memberi langkah konkret untuk mengerjakan bonus.

README ini **bukan** README final untuk pengumpulan. Nanti setelah implementasi selesai, README ini perlu diringkas menjadi README final yang lebih cocok untuk dosen/asisten.

## Sumber Konteks

Sumber analisis:

- PDF spesifikasi: `Spesifikasi Tugas Besar 2 IF3270 Pembelajaran Mesin.pdf`
- isi repository saat ini
- progres terbaru dari teman satu tim yang sudah masuk ke branch/worktree ini

Deadline pada PDF: **Jumat, 15 Mei 2026**

## Ringkasan Tugas

Tugas besar terdiri dari dua jalur utama:

- **CNN untuk image classification** pada dataset Intel Image Classification
- **Simple RNN dan LSTM untuk image captioning** pada dataset Flickr8k

Selain itu ada **bagian bonus** yang opsional.

Prioritas pengerjaan harus tetap:

1. selesaikan semua requirement wajib,
2. pastikan evaluasi dan deliverables utama lengkap,
3. baru pertimbangkan bonus.

## Requirement Wajib

### 1. CNN untuk Image Classification

Konteks:

- task: image classification
- dataset: Intel Image Classification
- jumlah kelas: 6
- split utama: train, validation, test

Yang diwajibkan:

- utility image loading berbasis PIL/Pillow dan NumPy
- feature extractor yang menyimpan output `.npy`
- training CNN menggunakan Keras
- implementasi forward propagation from scratch yang bisa membaca bobot Keras
- eksperimen hyperparameter CNN
- evaluasi macro F1-score
- perbandingan Keras vs scratch
- perbandingan shared parameter vs non-shared parameter

Layer CNN wajib:

- `Conv2D`
- `LocallyConnected2D`
- pooling layer
- `Flatten` atau global pooling
- `Dense`

Eksperimen CNN wajib:

- 2 variasi jumlah layer konvolusi
- 2 variasi kombinasi jumlah filter
- 2 variasi kombinasi ukuran kernel
- 2 variasi jenis pooling
- total **16 eksperimen**

Evaluasi CNN wajib:

- macro F1-score
- training/validation loss
- bandingkan shared vs non-shared
- bandingkan Keras vs scratch
- bandingkan jumlah parameter

### 2. RNN dan LSTM untuk Image Captioning

Konteks:

- task: image captioning
- dataset: Flickr8k
- sekitar 8092 gambar
- 5 caption per gambar
- split: 6000 train, 1000 validation, 1000 test

Arsitektur wajib:

- encoder-decoder
- encoder: CNN pretrained Keras yang frozen
- decoder: `SimpleRNN` dan `LSTM`
- metode injection: **pre-inject**

Komponen scratch wajib:

- `Embedding`
- `SimpleRNN cell`
- `LSTM cell`
- `Dense projection`
- `Dense output`

Yang diwajibkan:

- feature extraction CNN encoder ke `.npy`
- preprocessing caption
- training decoder Keras untuk RNN dan LSTM
- eksperimen jumlah layer dan hidden state
- implementasi decoder scratch
- pipeline end-to-end image ke caption
- evaluasi BLEU-4, METEOR, dan waktu eksekusi
- perbandingan RNN vs LSTM
- perbandingan Keras vs scratch
- eksperimen max caption length

Eksperimen captioning wajib:

- 3 variasi jumlah recurrent layer
- 2 variasi hidden state
- minimal 6 eksperimen untuk RNN
- minimal 6 eksperimen untuk LSTM
- total minimal **12 eksperimen**

### 3. Deliverables Wajib

Repository final minimal harus berisi:

- `src/`
- `doc/`
- `README.md`

Laporan final minimal harus mencakup:

- deskripsi persoalan
- penjelasan implementasi
- penjelasan forward propagation
- hasil pengujian CNN
- perbandingan shared vs non-shared
- analisis variasi hyperparameter CNN
- hasil pengujian image captioning
- perbandingan RNN vs LSTM
- perbandingan Keras vs scratch
- pengaruh max caption length
- kesimpulan
- pembagian tugas
- referensi

## Bonus

Bagian ini opsional.

Bonus dari spesifikasi:

- visualisasi feature map dan Grad-CAM
- captioning dengan arsitektur **init-inject**
- beam search decoder
- batch inference untuk seluruh forward scratch
- backward propagation from scratch

Bonus sebaiknya dikerjakan hanya setelah requirement wajib stabil.

## Kondisi Repository Saat Ini

Repository sekarang **tidak lagi kosong**. Jalur CNN sudah punya fondasi yang cukup jelas, sementara jalur captioning masih hampir kosong.

Struktur penting yang sudah ada:

- `src/utils/`
- `src/cnn/`
- `scripts/`
- `configs/cnn/base.json`
- dataset Intel ada di `dataset/`

Struktur penting yang belum terisi signifikan:

- `src/captioning/`
- `configs/captioning/`
- `doc/`

## Yang Sudah Selesai

Bagian ini berdasarkan file yang memang sudah ada saat README ini ditulis.

### 1. Utility umum

Sudah ada di `src/utils/`:

- image loading dasar
- batch image loading
- JSON/CSV writer
- training history CSV writer
- global seed helper

Artinya, fondasi utility umum sudah cukup baik untuk dipakai ulang di CNN maupun captioning.

### 2. Data pipeline dasar CNN

Sudah ada di `src/cnn/data.py`:

- `ImageRecord`
- `CNNDataset`
- scan folder labeled image
- scan folder unlabeled image
- stratified train/validation split
- konversi record ke array image dan label
- iterator batch berbasis NumPy loader
- hitung jumlah sample per kelas

Ini berarti temanmu sudah menyelesaikan:

- pembacaan struktur dataset Intel,
- class mapping berbasis urutan `class_names`,
- split train/validation dari `seg_train`,
- loading `seg_test`,
- loading `seg_pred`.

Status:

- **cukup bagus untuk baseline CNN Keras**
- **belum sama dengan feature extractor**
- **belum terkait scratch CNN**

### 3. Konfigurasi eksperimen CNN

Sudah ada di `src/cnn/config.py` dan `configs/cnn/base.json`.

Yang sudah tersedia:

- parsing config dari JSON
- validasi config
- validasi jumlah eksperimen = 16
- validasi layout dataset
- struktur config yang memisahkan:
- data
- training
- architecture defaults
- experiment grid
- output paths

Config `base.json` saat ini sudah memuat:

- dataset Intel
- image size `128x128`
- batch size `32`
- epochs `20`
- optimizer `adam`
- loss `sparse_categorical_crossentropy`
- comparison metric `macro_f1`
- conv layer count `[1, 2]`
- filter profiles `[[16, 32], [32, 64]]`
- kernel profiles `[[[3,3],[3,3]], [[5,5],[3,3]]]`
- pooling types `["max", "average"]`

Status:

- requirement grid eksperimen CNN **sudah dipersiapkan**
- jumlah eksperimen **sudah cocok 16**

### 4. Enumerasi eksperimen CNN

Sudah ada di `src/cnn/experiments.py`.

Yang sudah ada:

- `CNNExperiment`
- generator semua kombinasi eksperimen
- `run_id` yang konsisten
- lookup eksperimen berdasarkan `run_id`

Status:

- fondasi untuk automasi training 16 eksperimen **sudah ada**

### 5. Model CNN Keras shared parameter

Sudah ada di `src/cnn/keras_models.py`.

Yang sudah dibuat:

- builder model CNN shared berbasis `Conv2D`
- support pooling `max` dan `average`
- flatten
- dense hidden layer
- dense output
- compile dengan optimizer Adam

Status:

- baseline CNN Keras shared **sudah ada**
- ini sudah memenuhi sebagian requirement training model CNN shared
- **belum ada model non-shared `LocallyConnected2D`**

### 6. Training pipeline CNN Keras

Sudah ada di `src/cnn/training.py`.

Yang sudah dibuat:

- `CNNImageSequence`
- training satu eksperimen
- training banyak eksperimen
- skip run kalau artifact lengkap sudah ada
- evaluasi validation dengan:
- loss
- sparse categorical accuracy
- macro F1-score
- simpan artifact:
- `model.keras`
- `weights.weights.h5`
- `history.csv`
- `metrics.json`
- `experiment.json`
- `contract.json`
- export summary CSV

Status:

- pipeline training CNN shared Keras **sudah cukup jalan secara struktur**
- macro F1 validation **sudah ada**
- artifact management dasar **sudah ada**

Catatan:

- evaluasi yang tersimpan saat ini tampaknya fokus ke **validation sequence**, belum terlihat evaluasi final yang eksplisit untuk **test split**

### 7. Script CLI untuk CNN

Sudah ada di `scripts/`:

- `validate_cnn_config.py`
- `inspect_cnn_dataset.py`
- `train_cnn.py`

Kegunaannya:

- validasi config
- inspeksi dataset
- training eksperimen CNN

Status:

- jalur CNN sekarang sudah punya entry point CLI dasar

### 8. Dataset Intel sudah masuk repo/worktree

Folder dataset yang terdeteksi:

- `dataset/seg_train`
- `dataset/seg_test`
- `dataset/seg_pred`

Status:

- jalur CNN bisa dikerjakan langsung tanpa harus menunggu dataset Intel lagi

## Yang Belum Selesai

Bagian ini penting karena README sebelumnya terlalu menganggap repo masih kosong. Sekarang statusnya harus lebih presisi.

### A. CNN yang masih belum selesai

Meski jalur CNN shared Keras sudah punya fondasi, masih banyak requirement CNN yang belum selesai:

- belum ada feature extractor `.npy` berbasis CNN frozen
- belum ada evaluasi final yang eksplisit pada split **test**
- belum ada script analisis hasil 16 eksperimen
- belum ada plotting training/validation loss otomatis
- belum ada confusion matrix atau evaluasi klasifikasi lebih lengkap
- belum ada model `LocallyConnected2D` Keras untuk eksperimen non-shared
- belum ada perbandingan shared vs non-shared
- belum ada implementasi scratch untuk:
- `Conv2D`
- `LocallyConnected2D`
- pooling
- global pooling
- `Flatten`
- aktivasi
- `Dense`
- belum ada loader bobot Keras ke model scratch
- belum ada runner inference scratch
- belum ada pembanding Keras vs scratch

Kesimpulan jujur untuk CNN:

- **baseline training shared Keras sudah mulai terbentuk**
- **bagian scratch CNN belum mulai**
- **bagian non-shared juga belum mulai**

### B. Captioning yang masih belum selesai

Jalur captioning saat ini praktis masih kosong.

Belum ada:

- parser Flickr8k captions
- preprocessing caption
- vocabulary builder
- token mapping
- padding pipeline
- metadata max caption length
- feature extraction Flickr8k
- encoder frozen pipeline
- model Keras decoder RNN
- model Keras decoder LSTM
- eksperimen 12 variasi captioning
- evaluasi BLEU-4
- evaluasi METEOR
- benchmark waktu
- decoder scratch RNN
- decoder scratch LSTM
- pipeline autoregressive decoding
- qualitative analysis 10 contoh

### C. Deliverables yang masih belum selesai

Belum ada atau belum lengkap:

- `doc/` berisi laporan final
- README final untuk manusia
- pembagian tugas anggota
- notebook analisis formal
- laporan eksperimen yang siap dikumpulkan

## Ringkasan Status Saat Ini

Kalau dipetakan per area:

- utility umum: **sudah ada**
- data pipeline CNN: **sudah ada**
- config CNN: **sudah ada**
- eksperimen grid CNN: **sudah ada**
- model CNN shared Keras: **sudah ada**
- training CNN shared Keras: **sudah ada**
- evaluasi validation macro F1: **sudah ada**
- feature extractor CNN: **belum ada**
- CNN scratch: **belum ada**
- CNN non-shared: **belum ada**
- evaluasi test CNN lengkap: **belum jelas / belum lengkap**
- captioning pipeline: **belum ada**
- captioning scratch: **belum ada**
- deliverables laporan: **belum ada**

Secara keseluruhan:

- repo **sudah maju signifikan di jalur CNN shared Keras**
- repo **masih jauh dari selesai untuk seluruh spesifikasi**

## Cara Menjalankan Yang Sudah Ada

### 1. Validasi config CNN

```bash
python scripts/validate_cnn_config.py --config configs/cnn/base.json --check-data
```

Tujuan:

- memastikan config valid
- memastikan layout dataset sesuai config

### 2. Inspeksi dataset CNN

```bash
python scripts/inspect_cnn_dataset.py --config configs/cnn/base.json --sample-batch
```

Tujuan:

- melihat jumlah data train/validation/test
- cek distribusi kelas
- cek shape batch sample

### 3. Dry run eksperimen CNN

```bash
python scripts/train_cnn.py --config configs/cnn/base.json --dry-run
```

Tujuan:

- melihat daftar `run_id`
- melihat jumlah parameter model untuk tiap eksperimen

### 4. Train sebagian eksperimen CNN

```bash
python scripts/train_cnn.py --config configs/cnn/base.json --limit 2
```

### 5. Train run tertentu

```bash
python scripts/train_cnn.py --config configs/cnn/base.json --run-id shared_l1_f16_k3x3_max
```

Catatan:

- format `run_id` bergantung hasil generator eksperimen
- pakai `--dry-run` dulu untuk memastikan nama run

## Apa yang Harus Dilakukan Selanjutnya

Urutan di bawah ini disusun berdasarkan status repo saat ini, jadi tidak lagi mengulang dari nol.

### Prioritas 1: Rapikan dan stabilkan jalur CNN yang sudah ada

Sebelum lompat ke captioning, jalur CNN yang sudah dibangun perlu diselesaikan sampai benar-benar memenuhi spesifikasi shared Keras.

Langkah konkret:

1. tambahkan evaluasi **test split** resmi, bukan hanya validation
2. tambahkan export prediction CSV/JSON untuk test split
3. tambahkan plotting history training/validation loss per run
4. tambahkan summary ranking 16 eksperimen berdasarkan macro F1 test
5. tentukan eksperimen terbaik shared Keras

Output yang diinginkan:

- satu tabel ranking 16 eksperimen
- satu kandidat model terbaik shared Keras

### Prioritas 2: Kerjakan eksperimen non-shared CNN

Setelah shared Keras beres, lanjut ke requirement non-shared.

Langkah konkret:

1. buat builder model Keras dengan `LocallyConnected2D`
2. buat konfigurasi eksperimen non-shared yang setara dengan arsitektur terbaik shared
3. train model non-shared
4. evaluasi macro F1
5. bandingkan jumlah parameter
6. bandingkan loss curve
7. tulis analisis efisiensi dan performa

### Prioritas 3: Implementasi CNN scratch

Ini requirement besar yang belum dikerjakan.

Urutan implementasi yang disarankan:

1. `Dense`
2. `Flatten`
3. `ReLU`
4. `Softmax`
5. `MaxPooling2D`
6. `AveragePooling2D`
7. `GlobalAveragePooling2D`
8. `GlobalMaxPooling2D`
9. `Conv2D`
10. `LocallyConnected2D`

Setelah layer tersedia:

1. buat loader bobot dari model Keras
2. buat runner forward model scratch
3. validasi output scratch vs Keras di beberapa sample kecil
4. jalankan evaluasi test penuh
5. hitung macro F1 scratch
6. bandingkan Keras vs scratch

### Prioritas 4: Tambahkan feature extractor CNN

Walau spesifikasi menyebut feature extractor pada bagian CNN utility, implementasi ini juga akan sangat membantu captioning.

Yang perlu dibuat:

- load encoder CNN frozen
- ekstraksi feature batch
- simpan `.npy`
- simpan mapping file ke feature

Saran:

- letakkan di jalur terpisah, misalnya `src/cnn/features.py`
- buat script CLI khusus

### Prioritas 5: Mulai jalur captioning dari preprocessing dan feature extraction

Jangan mulai dari decoder scratch dulu. Mulai dari data dan feature extraction.

Urutan yang disarankan:

1. parsing caption Flickr8k
2. cleaning text
3. tokenisasi
4. vocabulary
5. padding
6. metadata max length
7. feature extraction image dengan encoder frozen

### Prioritas 6: Training decoder Keras captioning

Setelah preprocessing dan feature extraction siap:

1. buat builder decoder `SimpleRNN`
2. buat builder decoder `LSTM`
3. implement teacher forcing pre-inject
4. jalankan 6 eksperimen RNN
5. jalankan 6 eksperimen LSTM
6. simpan history, metric, bobot, dan waktu

### Prioritas 7: Scratch decoder captioning

Setelah baseline Keras captioning stabil:

1. `Embedding`
2. `DenseProjection`
3. `DenseOutput`
4. `SimpleRNNCell`
5. `LSTMCell`
6. greedy decoding
7. evaluasi BLEU-4, METEOR, waktu

## Checklist Requirement Wajib

### CNN

- [x] utility image dasar
- [x] data pipeline Intel dataset
- [x] config eksperimen CNN
- [x] generator 16 eksperimen CNN
- [x] builder CNN shared Keras
- [x] training pipeline CNN shared Keras
- [x] evaluasi validation macro F1
- [ ] evaluasi test split resmi
- [ ] feature extractor `.npy`
- [ ] builder CNN non-shared Keras
- [ ] eksperimen shared vs non-shared
- [ ] implementasi scratch CNN
- [ ] pembanding Keras vs scratch

### Captioning

- [ ] preprocessing Flickr8k
- [ ] feature extraction Flickr8k
- [ ] decoder Keras RNN
- [ ] decoder Keras LSTM
- [ ] 12 eksperimen captioning
- [ ] decoder scratch RNN
- [ ] decoder scratch LSTM
- [ ] BLEU-4
- [ ] METEOR
- [ ] benchmark waktu
- [ ] qualitative analysis
- [ ] eksperimen max caption length

### Deliverables

- [ ] folder `doc` final
- [ ] laporan PDF final
- [ ] README final untuk pengumpulan
- [ ] pembagian tugas anggota

## Cara Mengerjakan Bonus

Bagian ini sengaja dibuat lebih operasional supaya AI lain bisa langsung lanjut.

### 1. Bonus Grad-CAM dan Visualisasi Feature Map

Tujuan:

- menunjukkan region gambar yang paling berpengaruh pada prediksi CNN
- membantu analisis model CNN

Langkah kerja:

1. pilih model CNN shared terbaik hasil eksperimen
2. identifikasi layer konvolusi terakhir
3. buat fungsi untuk mengambil activation map layer tersebut
4. untuk **feature map visualization**, tampilkan beberapa channel activation sebagai grid gambar
5. untuk **Grad-CAM**, hitung gradient skor kelas target terhadap activation map layer terakhir
6. lakukan global average pooling terhadap gradient
7. bobotkan activation map dengan gradient hasil pooling
8. ambil `ReLU` hasil akhirnya
9. resize heatmap ke ukuran gambar asli
10. overlay heatmap ke gambar input
11. simpan hasil visualisasi untuk beberapa contoh benar dan salah klasifikasi

Output yang disarankan:

- folder `reports/cnn/gradcam/`
- folder `reports/cnn/feature_maps/`

### 2. Bonus Captioning Init-Inject

Tujuan:

- membandingkan arsitektur **pre-inject** wajib dengan **init-inject** bonus

Konsep:

- pada pre-inject, feature CNN menjadi input timestep sebelum `<start>`
- pada init-inject, feature gambar dipakai untuk menginisialisasi hidden state atau digabung setelah decoder memproses prefix, sesuai variasi yang dipilih

Langkah kerja:

1. pertahankan pipeline preprocessing dan feature extraction yang sama
2. buat builder decoder alternatif untuk init-inject
3. putuskan mekanisme injeksi:
- opsi A: feature diproyeksikan untuk menjadi hidden state awal
- opsi B: feature digabung dengan representasi caption setelah recurrent stack
4. latih dengan konfigurasi eksperimen yang sama seperti pre-inject terbaik
5. evaluasi BLEU-4, METEOR, dan waktu
6. bandingkan hasil dengan arsitektur pre-inject wajib
7. analisis apakah image information lebih efektif dimasukkan di awal hidden state atau di input sequence

Output yang disarankan:

- tabel perbandingan pre-inject vs init-inject

### 3. Bonus Beam Search Decoder

Tujuan:

- mengganti greedy decoding dengan pencarian caption yang lebih baik

Langkah kerja:

1. pastikan greedy decoding captioning sudah benar dulu
2. buat fungsi beam search dengan parameter `beam_width`, misalnya `3` atau `5`
3. pada tiap timestep:
- simpan beberapa partial caption terbaik
- perluas masing-masing partial caption dengan token kandidat teratas
- hitung skor akumulatif log probability
4. gunakan normalisasi panjang jika hasil terlalu bias ke caption pendek
5. stop saat semua beam selesai dengan token `<end>` atau mencapai `max_length`
6. bandingkan hasil beam search dengan greedy pada subset test yang sama
7. catat BLEU-4, METEOR, dan waktu tambahan

Output yang disarankan:

- tabel `greedy vs beam search`
- beberapa contoh caption yang membaik dan yang memburuk

### 4. Bonus Batch Inference Scratch

Tujuan:

- membuat seluruh forward scratch menerima batch input, bukan cuma satu sampel

Langkah kerja:

1. audit semua layer scratch agar shape input jelas
2. tetapkan konvensi tensor:
- CNN: `NHWC`
- sequence: `batch, time, feature`
3. ubah operasi yang masih scalar/per-sample menjadi operasi batch
4. verifikasi hasil batch size `1` sama dengan mode single sample
5. uji batch size `2`, `4`, `8`
6. cek apakah output tetap identik secara numerik terhadap Keras

Catatan:

- bonus ini sangat berguna karena sekaligus membuat implementasi scratch lebih rapi dan bisa dipakai untuk evaluasi lebih cepat

### 5. Bonus Backward Propagation Scratch

Tujuan:

- melatih model atau minimal membuktikan turunan backward untuk layer scratch

Langkah kerja:

1. simpan cache forward untuk setiap layer
2. implementasikan backward untuk layer paling sederhana dulu:
- Dense
- ReLU
- Flatten
3. lanjut ke pooling
4. lanjut ke Conv2D
5. jika waktu cukup, lanjut ke recurrent cells
6. buat gradient check numerik pada contoh tensor kecil
7. cocokkan gradient scratch dengan finite difference approximation

Catatan:

- ini bonus paling berat
- jangan mulai dari sini sebelum requirement wajib benar-benar aman

## Risiko dan Titik Sulit

Titik sulit utama:

- `LocallyConnected2D` scratch
- menjaga konsistensi shape dan format bobot Keras
- membangun evaluasi test CNN yang rapi di atas pipeline training sekarang
- menyusun arsitektur captioning pre-inject dengan benar
- membuat decoder scratch autoregressive
- mengelola banyak artifact eksperimen

## Saran untuk AI Lain

- jangan reset atau hapus perubahan teman tanpa alasan
- lanjutkan fondasi CNN yang sudah ada, jangan bangun ulang dari nol
- selesaikan shared Keras sampai evaluasi test dan ranking eksperimen beres
- setelah itu baru non-shared dan scratch CNN
- jangan masuk ke captioning sebelum jalur CNN shared benar-benar stabil
- simpan semua artifact eksperimen
- buat script kecil untuk evaluasi dan plotting, jangan menumpuk semua logic di notebook

## Catatan Penutup

Repository ini sekarang berada pada fase:

- **CNN shared Keras sudah mulai matang secara fondasi**
- **CNN non-shared dan scratch belum ada**
- **captioning hampir belum mulai**

Strategi terbaik dari titik ini:

1. selesaikan evaluasi dan eksperimen CNN shared
2. kerjakan CNN non-shared
3. kerjakan CNN scratch
4. baru masuk captioning
5. terakhir, tambah bonus jika waktu masih cukup
