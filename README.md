# Tugas Besar 2 IF3270 Pembelajaran Mesin

## Tujuan Repository

Repository ini dipakai untuk mengerjakan Tugas Besar 2 IF3270 Pembelajaran Mesin dengan fokus pada dua bagian utama:

- implementasi dan eksperimen Convolutional Neural Network (CNN) untuk image classification,
- implementasi dan eksperimen Simple RNN dan LSTM untuk image captioning.

Dokumen ini sengaja ditulis sebagai handoff context untuk AI lain atau anggota tim lain. Targetnya adalah supaya siapa pun yang melanjutkan repository ini bisa langsung paham:

- apa isi spesifikasi tugas,
- mana requirement wajib,
- mana bonus,
- kondisi repository saat ini,
- apa yang sudah selesai,
- apa yang belum selesai,
- dan urutan kerja yang paling masuk akal.

## Sumber Konteks

Analisis README ini disusun dari:

- PDF spesifikasi: `Spesifikasi Tugas Besar 2 IF3270 Pembelajaran Mesin.pdf`
- isi repository saat ini.

Deadline pada PDF: **Jumat, 15 Mei 2026**.

## Ringkasan Spesifikasi

Tugas besar dibagi menjadi dua jalur utama:

- CNN untuk image classification.
- RNN/LSTM untuk image captioning.

Ada juga bagian bonus, tetapi bonus bersifat opsional. Fokus utama repository ini harus tetap pada pemenuhan requirement wajib terlebih dahulu.

## Requirement Wajib

### 1. Bagian CNN

Konteks:

- Task: image classification.
- Dataset: Intel Image Classification.
- Jumlah kelas: 6.
- Split train, validation, dan test sudah tersedia dari dataset.

#### 1.1 Utility dasar image

Harus ada utility function berbasis PIL/Pillow dan NumPy, tanpa Keras preprocessing untuk bagian image utility dasar.

Yang diwajibkan:

- image loader dari file path,
- resize ke ukuran target,
- konversi ke NumPy array,
- normalisasi pixel ke rentang `[0, 1]`,
- batch loader untuk list path menjadi tensor `(N, H, W, C)`.

#### 1.2 Feature extractor

Harus ada utility feature extractor yang:

- menerima list path gambar,
- menggunakan CNN encoder Keras yang frozen,
- mengekstraksi feature vector,
- menyimpan hasilnya ke disk dalam format `.npy` agar tidak diekstraksi ulang.

#### 1.3 Training model CNN dengan Keras

Harus melatih model CNN menggunakan Keras.

Layer minimal yang harus tercakup pada arsitektur CNN:

- `Conv2D` shared parameter,
- pooling layer,
- `Flatten` atau global pooling,
- `Dense`.

Loss dan optimizer yang diwajibkan:

- loss: `SparseCategoricalCrossentropy`,
- optimizer: `Adam`.

#### 1.4 Forward propagation CNN from scratch

Harus mengimplementasikan forward propagation from scratch yang dapat membaca bobot hasil training Keras.

Setiap layer sebaiknya modular dan punya method `forward(...)`.

Layer wajib:

- `Conv2D` shared parameters,
- `LocallyConnected2D` non-shared parameters,
- `MaxPooling2D` dan/atau `AveragePooling2D`,
- `GlobalAveragePooling2D` dan/atau `GlobalMaxPooling2D`,
- `Flatten`,
- fungsi aktivasi seperti `ReLU` dan `Softmax`,
- `Dense`.

Catatan dari spesifikasi:

- Dense boleh mengadaptasi implementasi FFNN dari Tubes 1.

#### 1.5 Eksperimen CNN

Harus melakukan variasi hyperparameter berikut untuk model CNN shared parameter:

- jumlah layer konvolusi: 2 variasi,
- banyak filter per layer: 2 variasi kombinasi,
- ukuran filter per layer: 2 variasi kombinasi,
- jenis pooling: 2 variasi, yaitu max pooling dan average pooling.

Total eksperimen yang diharapkan: **16 arsitektur**.

#### 1.6 Evaluasi CNN

Metrik utama yang diwajibkan:

- **macro F1-score**.

Yang harus dilakukan:

- simpan bobot semua model hasil training,
- pilih arsitektur terbaik dari eksperimen Keras,
- jalankan forward propagation from scratch untuk arsitektur shared,
- buat variasi arsitektur non-shared dengan mengganti semua `Conv2D` menjadi `LocallyConnected2D`,
- bandingkan Keras vs scratch,
- bandingkan shared vs non-shared,
- bandingkan jumlah parameter,
- bandingkan training/validation loss,
- dan tulis analisis kesimpulannya.

### 2. Bagian RNN dan LSTM untuk Image Captioning

Konteks:

- Task: image captioning.
- Dataset: Flickr8k.
- Sekitar 8.092 gambar.
- Tiap gambar punya 5 caption.
- Split: 6000 train, 1000 validation, 1000 test.

Arsitektur wajib:

- encoder-decoder,
- encoder: CNN pretrained Keras yang frozen,
- decoder: dua model terpisah, yaitu `SimpleRNN` dan `LSTM`.

Metode injection yang diwajibkan:

- **pre-inject**.

Artinya:

- feature vector dari CNN diproyeksikan ke `embed_dim` melalui Dense,
- hasil proyeksi itu dipakai sebagai input timestep sebelum token `<start>`,
- hidden state awal bernilai nol,
- untuk LSTM, cell state awal juga nol.

#### 2.1 Forward propagation from scratch untuk decoder

Komponen wajib yang harus diimplementasikan dari nol:

- `Embedding` layer,
- `SimpleRNN cell`,
- `LSTM cell`,
- `Dense projection layer`,
- `Dense output layer` dengan softmax.

#### 2.2 Feature extraction CNN encoder

Harus menggunakan CNN pretrained dari Keras, tanpa classification head, dan bobot ImageNet yang dibekukan.

Direkomendasikan oleh spesifikasi:

- `InceptionV3`, atau
- `VGG16`.

Yang harus dilakukan:

- forward pass seluruh gambar Flickr8k,
- simpan feature vectors ke disk dalam format `.npy`,
- hasil ini dipakai ulang untuk training decoder RNN dan LSTM.

#### 2.3 Preprocessing caption

Yang diwajibkan:

- lowercase,
- hapus tanda baca,
- tokenisasi,
- bangun vocabulary dari caption training,
- token khusus wajib:
- `<start>`
- `<end>`
- `<pad>`
- padding sequence ke panjang seragam,
- simpan vocabulary ke disk, misalnya `.json`.

#### 2.4 Training decoder Keras

Harus melatih dua jenis decoder secara terpisah:

- `SimpleRNN`,
- `LSTM`.

Variasi training yang diwajibkan untuk masing-masing decoder:

- jumlah layer recurrent: 3 variasi,
- hidden state size: 2 variasi.

Minimum total eksperimen:

- 6 variasi untuk RNN,
- 6 variasi untuk LSTM,
- total 12 variasi.

#### 2.5 Implementasi arsitektur end-to-end

Harus bisa load:

- bobot CNN encoder pretrained,
- bobot decoder hasil training,
- lalu menjalankan pipeline dari raw image sampai menghasilkan caption.

Harus ada 2 arsitektur scratch utama:

- encoder + decoder RNN,
- encoder + decoder LSTM.

#### 2.6 Evaluasi captioning

Yang diwajibkan:

- catat **BLEU-4**,
- catat **waktu eksekusi**,
- pilih 1 variasi terbaik untuk masing-masing RNN dan LSTM,
- bandingkan dengan implementasi Keras yang setara,
- pilih arsitektur terbaik dari kombinasi RNN/LSTM dan Keras/Scratch,
- lakukan variasi panjang maksimum caption minimal 3 variasi,
- catat pengaruhnya terhadap score.

Evaluasi dan analisis yang diwajibkan pada laporan:

- perbandingan jumlah layer dan hidden state,
- perbandingan RNN vs LSTM,
- perbandingan Keras vs scratch,
- BLEU-4 dan METEOR,
- training loss dan validation loss,
- analisis kualitatif minimal 10 contoh gambar dengan caption ground truth dan caption hasil model,
- analisis konsep vanishing gradient dan memori jangka panjang untuk menjelaskan perbedaan RNN dan LSTM.

### 3. Deliverables Wajib Repository

Repository final minimal harus berisi:

- folder `src`,
- folder `doc`,
- `README.md`.

Isi minimal `README.md` final untuk manusia:

- deskripsi singkat repository,
- cara setup,
- cara menjalankan program,
- pembagian tugas tiap anggota kelompok.

Isi minimal laporan PDF di folder `doc`:

- cover,
- deskripsi persoalan,
- pembahasan,
- penjelasan implementasi,
- penjelasan forward propagation,
- hasil pengujian CNN,
- perbandingan shared vs non-shared,
- pengaruh jumlah layer/filter/kernel/pooling pada CNN,
- hasil pengujian image captioning,
- perbandingan jumlah layer untuk RNN dan LSTM,
- perbandingan hidden state untuk RNN dan LSTM,
- perbandingan RNN vs LSTM,
- perbandingan Keras vs scratch,
- pengaruh panjang maksimum caption terhadap BLEU-4,
- kesimpulan dan saran,
- pembagian tugas,
- referensi,
- lampiran form penggunaan AI.

## Bagian Bonus

Bagian ini **opsional**, bukan requirement minimum.

Bonus yang tercantum pada spesifikasi:

- `[CNN]` visualisasi fitur intermediate dan Grad-CAM.
- `[RNN/LSTM]` image captioning dengan arsitektur alternatif **init-inject**.
- `[RNN/LSTM]` beam search decoder, misalnya `k=3` atau `k=5`.
- `[Semua]` batch inference pada seluruh implementasi forward propagation from scratch.
- `[Semua]` backward propagation from scratch.

Catatan penting:

- Bonus **tidak boleh mengganggu penyelesaian requirement wajib**.
- Jika waktu terbatas, jangan sentuh bonus sebelum semua requirement inti selesai.
- Dari semua bonus, yang paling realistis untuk ditambahkan setelah requirement inti selesai biasanya adalah:
- batch inference,
- beam search,
- Grad-CAM.

## Kondisi Repository Saat Ini

Saat file ini ditulis, repository masih dalam tahap awal.

Struktur file yang terdeteksi:

- `src/__init__.py`
- `src/utils/__init__.py`
- `src/utils/images.py`
- `src/utils/io.py`
- `src/utils/random.py`
- `src/captioning/__init__.py`
- `configs/captioning/.gitkeep`
- `requirements.txt`

README lama sebelumnya hampir kosong.

## Yang Sudah Selesai

### 1. Utility image loading dasar

File: `src/utils/images.py`

Sudah ada:

- `load_image(...)`
- `load_image_batch(...)`

Detail yang sudah terpenuhi:

- load gambar dengan `PIL.Image.open`,
- convert mode warna,
- resize ke target size,
- konversi ke NumPy array,
- normalisasi ke `[0, 1]`,
- validasi shape,
- dukungan batch image dari list path ke tensor `(N, H, W, C)`.

Status terhadap spesifikasi:

- ini sudah memenuhi sebagian requirement utility dasar image untuk bagian CNN,
- dan bisa dipakai ulang untuk preprocessing image pada captioning.

### 2. Utility I/O dasar

File: `src/utils/io.py`

Sudah ada:

- `read_json(...)`
- `write_json(...)`
- `write_csv(...)`
- `write_history_csv(...)`

Status:

- sudah berguna untuk menyimpan konfigurasi,
- menyimpan hasil eksperimen,
- menyimpan history training,
- dan menyimpan vocabulary caption.

### 3. Utility reproducibility

File: `src/utils/random.py`

Sudah ada:

- `set_global_seed(...)`

Status:

- ini membantu reproducibility untuk `random`, `numpy`, dan `tensorflow`.

### 4. Re-export utility

File: `src/utils/__init__.py`

Sudah ada ekspor utilitas agar impor modul lebih rapi.

## Yang Belum Selesai

Bagian inti tugas hampir semuanya belum ada di repository saat ini.

### A. Yang belum ada untuk CNN

- belum ada data pipeline Intel Image Classification,
- belum ada class mapping dan metadata dataset,
- belum ada builder model CNN Keras,
- belum ada script training CNN,
- belum ada 16 eksperimen hyperparameter CNN,
- belum ada penyimpanan bobot seluruh model eksperimen,
- belum ada evaluasi macro F1-score,
- belum ada plotting training/validation loss,
- belum ada feature extractor `.npy`,
- belum ada implementasi `Conv2D` scratch,
- belum ada implementasi `LocallyConnected2D` scratch,
- belum ada implementasi pooling scratch,
- belum ada implementasi global pooling scratch,
- belum ada implementasi `Flatten` scratch,
- belum ada implementasi aktivasi scratch,
- belum ada implementasi `Dense` scratch di repo ini,
- belum ada adaptor load bobot Keras ke scratch,
- belum ada script pembanding Keras vs scratch,
- belum ada eksperimen shared vs non-shared.

### B. Yang belum ada untuk captioning

- belum ada parser caption Flickr8k,
- belum ada preprocessing teks,
- belum ada vocabulary builder,
- belum ada token mapping,
- belum ada padding pipeline,
- belum ada metadata max caption length,
- belum ada feature extractor Flickr8k dengan CNN frozen,
- belum ada penyimpanan feature `.npy` untuk Flickr8k,
- belum ada builder decoder Keras untuk RNN,
- belum ada builder decoder Keras untuk LSTM,
- belum ada eksperimen 6 variasi RNN,
- belum ada eksperimen 6 variasi LSTM,
- belum ada implementasi `Embedding` scratch,
- belum ada implementasi `SimpleRNNCell` scratch,
- belum ada implementasi `LSTMCell` scratch,
- belum ada dense projection scratch,
- belum ada dense output scratch,
- belum ada greedy decoding pipeline,
- belum ada evaluasi BLEU-4,
- belum ada evaluasi METEOR,
- belum ada benchmark waktu eksekusi,
- belum ada qualitative analysis 10 contoh gambar.

### C. Yang belum ada untuk deliverables

- belum ada folder `doc` berisi laporan PDF,
- belum ada notebook pengujian,
- belum ada dokumentasi run pipeline,
- belum ada pembagian tugas anggota kelompok pada README final untuk manusia.

## Kesimpulan Status Repository

Status repo saat ini masih setara dengan fase bootstrap awal.

Secara praktis:

- fondasi utility dasar sudah mulai ada,
- implementasi inti model belum ada,
- training belum ada,
- eksperimen belum ada,
- evaluasi belum ada,
- laporan belum ada.

Kalau diukur terhadap spesifikasi keseluruhan, progress masih rendah dan mayoritas pekerjaan utama masih tersisa.

## Apa yang Harus Dilakukan Selanjutnya

Bagian ini ditulis sebagai arahan kerja untuk AI lain. Fokusnya adalah langkah yang paling rasional, berurutan, dan minim dead-end.

### Prioritas 1: Rapikan Struktur Proyek

Sebelum menambah banyak kode, rapikan struktur folder dulu.

Struktur yang disarankan:

- `src/cnn/`
- `src/cnn/keras/`
- `src/cnn/scratch/`
- `src/cnn/experiments/`
- `src/captioning/preprocessing/`
- `src/captioning/encoder/`
- `src/captioning/keras/`
- `src/captioning/scratch/`
- `src/captioning/experiments/`
- `src/metrics/`
- `configs/cnn/`
- `configs/captioning/`
- `artifacts/models/`
- `artifacts/features/`
- `artifacts/histories/`
- `artifacts/results/`
- `notebooks/`
- `doc/`

Tujuan langkah ini:

- memisahkan training Keras dari implementasi scratch,
- mencegah file terlalu besar,
- memudahkan AI lain bernavigasi,
- memudahkan penyimpanan artifact eksperimen.

### Prioritas 2: Selesaikan Jalur CNN Terlebih Dahulu

Pipeline CNN lebih cocok dijadikan tahap pertama karena:

- scope-nya lebih jelas,
- evaluasinya lebih sederhana daripada captioning,
- sebagian utility yang dibuat bisa dipakai ulang.

#### 2.1 Bangun data pipeline Intel dataset

Yang perlu dibuat:

- pembaca struktur direktori dataset,
- list file path untuk train/validation/test,
- mapping nama kelas ke integer,
- helper batching,
- metadata dataset ke JSON.

Output yang diharapkan:

- list sample train/val/test,
- label mapping,
- fungsi loader yang reusable.

#### 2.2 Bangun training CNN Keras

Yang perlu dibuat:

- builder arsitektur CNN berbasis konfigurasi,
- fungsi training,
- fungsi evaluasi,
- fungsi save model, save weights, save history.

Hyperparameter yang harus bisa divariasikan:

- jumlah conv layer,
- jumlah filter,
- ukuran kernel,
- jenis pooling.

Target minimum:

- bisa menjalankan **16 eksperimen**.

Setiap eksperimen sebaiknya menyimpan:

- nama run,
- config,
- history,
- bobot/model,
- prediksi test,
- metrik akhir.

#### 2.3 Implementasi macro F1 dan plotting

Yang perlu dibuat:

- evaluasi `macro F1-score`,
- opsi confusion matrix,
- plotting training loss dan validation loss.

#### 2.4 Implementasi feature extractor CNN

Yang perlu dibuat:

- load encoder Keras frozen,
- ekstraksi feature per batch,
- simpan `.npy`,
- simpan mapping antara file dan feature.

Walau feature extractor ini muncul di requirement CNN, implementasi ini juga akan sangat membantu jalur captioning.

#### 2.5 Implementasi forward propagation CNN from scratch

Urutan implementasi yang disarankan:

1. `Dense`
2. `Flatten`
3. `ReLU`
4. `Softmax`
5. pooling
6. global pooling
7. `Conv2D`
8. `LocallyConnected2D`

Kebutuhan desain:

- semua berbasis NumPy,
- setiap layer modular,
- mudah load bobot dari Keras,
- shape konsisten dengan Keras,
- sebisa mungkin mendukung batch inference.

#### 2.6 Validasi Keras vs scratch

Yang perlu dibuat:

- loader model/bobot terbaik dari eksperimen CNN,
- adaptor model Keras ke representasi scratch,
- script inference pada split test,
- perbandingan prediksi dan macro F1 antara Keras dan scratch.

#### 2.7 Eksperimen shared vs non-shared

Yang perlu dibuat:

- arsitektur non-shared dengan `LocallyConnected2D`,
- training/evaluasi,
- hitung jumlah parameter,
- bandingkan performa dan efisiensi.

### Prioritas 3: Kerjakan Jalur Captioning

Setelah CNN stabil, baru lanjut ke captioning.

#### 3.1 Bangun preprocessing caption Flickr8k

Yang perlu dibuat:

- parser caption file,
- lowercase,
- hapus tanda baca,
- tambah token `<start>` dan `<end>`,
- bangun vocabulary dari train set,
- token to id,
- id to token,
- padding sequence,
- simpan vocabulary dan metadata.

Output penting yang disarankan:

- `vocab.json`,
- `token_to_id.json`,
- `id_to_token.json`,
- `caption_metadata.json`,
- file sequence numerik untuk training.

#### 3.2 Ekstraksi feature Flickr8k

Yang perlu dibuat:

- pilih backbone pretrained, misalnya `InceptionV3` atau `VGG16`,
- hapus head klasifikasi,
- freeze bobot,
- ekstrak feature seluruh image,
- simpan ke `.npy`,
- simpan pemetaan image id ke feature.

#### 3.3 Training decoder Keras untuk RNN dan LSTM

Yang perlu dibuat:

- builder decoder `SimpleRNN`,
- builder decoder `LSTM`,
- support recurrent layers yang bervariasi,
- support hidden size yang bervariasi,
- implement teacher forcing sesuai spesifikasi pre-inject.

Semua eksperimen harus menyimpan:

- config,
- history,
- weights,
- metric,
- waktu eksekusi.

#### 3.4 Implementasi scratch decoder

Yang perlu dibuat:

- `Embedding`,
- `DenseProjection`,
- `DenseOutput`,
- `SimpleRNNCell`,
- `LSTMCell`,
- wrapper decoder sequence,
- greedy decoding.

Urutan implementasi yang disarankan:

1. `Embedding`
2. `DenseProjection`
3. `DenseOutput`
4. `SimpleRNNCell`
5. `LSTMCell`
6. sequence wrapper
7. inference autoregressive

#### 3.5 Implementasi pipeline raw image ke caption

Yang perlu dibuat:

- load dan preprocess image,
- ekstrak feature dengan encoder,
- project feature ke embedding space,
- decode token demi token,
- stop di `<end>` atau `max_length`.

Minimal ada dua pipeline scratch:

- image -> encoder -> RNN decoder,
- image -> encoder -> LSTM decoder.

#### 3.6 Evaluasi captioning

Yang perlu dibuat:

- BLEU-4,
- METEOR,
- waktu eksekusi,
- penyimpanan caption hasil prediksi,
- analisis kualitatif 10 contoh.

#### 3.7 Eksperimen max caption length

Yang perlu dibuat:

- pilih arsitektur terbaik dari empat kelompok:
- RNN scratch,
- LSTM scratch,
- RNN Keras,
- LSTM Keras,
- lalu variasikan max caption length minimal 3 nilai,
- ukur pengaruh terhadap BLEU-4.

## Checklist Requirement Minimum

AI lain bisa memakai checklist ini untuk memastikan requirement wajib sudah terpenuhi.

### Checklist CNN

- [ ] data pipeline Intel dataset selesai
- [ ] utility image dasar siap dipakai
- [ ] feature extractor `.npy` tersedia
- [ ] builder CNN Keras tersedia
- [ ] 16 eksperimen CNN selesai
- [ ] history dan weights semua eksperimen tersimpan
- [ ] evaluasi macro F1 tersedia
- [ ] implementasi `Conv2D` scratch selesai
- [ ] implementasi `LocallyConnected2D` scratch selesai
- [ ] implementasi pooling/global pooling/flatten/dense/aktivasi scratch selesai
- [ ] pembanding Keras vs scratch selesai
- [ ] pembanding shared vs non-shared selesai

### Checklist Captioning

- [ ] preprocessing caption Flickr8k selesai
- [ ] vocabulary dan metadata caption tersimpan
- [ ] feature extraction Flickr8k selesai
- [ ] 6 eksperimen RNN Keras selesai
- [ ] 6 eksperimen LSTM Keras selesai
- [ ] implementasi `Embedding` scratch selesai
- [ ] implementasi `SimpleRNNCell` scratch selesai
- [ ] implementasi `LSTMCell` scratch selesai
- [ ] dense projection dan dense output scratch selesai
- [ ] pipeline caption generation scratch selesai
- [ ] BLEU-4 tersedia
- [ ] METEOR tersedia
- [ ] benchmark waktu eksekusi tersedia
- [ ] analisis kualitatif minimal 10 contoh tersedia
- [ ] eksperimen max caption length selesai

### Checklist Deliverables

- [ ] folder `src` lengkap
- [ ] folder `doc` ada
- [ ] laporan PDF ada
- [ ] README final untuk manusia ada
- [ ] pembagian tugas anggota ada

## Bonus Checklist

Bagian ini opsional. Kerjakan hanya jika requirement wajib sudah aman.

- [ ] Grad-CAM / visualisasi feature map
- [ ] captioning init-inject
- [ ] beam search decoder
- [ ] batch inference scratch umum
- [ ] backward propagation scratch

## Risiko dan Titik Sulit

Bagian yang kemungkinan paling sulit:

- `LocallyConnected2D` scratch,
- konsistensi shape dan format bobot Keras,
- validasi numerik antara Keras dan scratch,
- implementasi pre-inject yang benar untuk captioning,
- decoding autoregressive RNN/LSTM scratch,
- manajemen artifact eksperimen yang banyak,
- waktu training yang besar jika eksperimen tidak diotomasi.

## Saran untuk AI Lain

- Jangan langsung lompat ke bonus.
- Jangan mulai dari notebook besar yang mencampur semua hal.
- Bangun modul Python yang terpisah untuk preprocessing, training, scratch, dan evaluasi.
- Simpan semua artifact eksperimen agar tidak perlu training ulang.
- Uji layer scratch secara kecil sebelum mencoba pipeline penuh.
- Selesaikan CNN sampai stabil, lalu baru captioning.
- Pastikan setiap eksperimen punya config dan output yang terdokumentasi.

## Ringkasan Singkat Status Saat Ini

Status current repo:

- utility image dasar: **sudah ada**
- utility I/O: **sudah ada**
- utility seed: **sudah ada**
- training CNN Keras: **belum ada**
- CNN scratch: **belum ada**
- eksperimen CNN: **belum ada**
- preprocessing caption: **belum ada**
- feature extraction captioning: **belum ada**
- decoder Keras RNN/LSTM: **belum ada**
- decoder scratch RNN/LSTM: **belum ada**
- evaluasi F1/BLEU/METEOR: **belum ada**
- laporan dan dokumentasi final: **belum ada**

## Catatan Penutup

Repository ini baru punya fondasi awal berupa utility. Hampir semua bagian inti tugas besar masih harus dibangun.

Kalau AI lain mengambil alih dari titik ini, strategi terbaik adalah:

1. rapikan struktur repo,
2. selesaikan seluruh jalur CNN wajib,
3. lanjut ke seluruh jalur captioning wajib,
4. baru pertimbangkan bonus,
5. terakhir rapikan laporan dan dokumentasi final.
