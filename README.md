# AAS_ComputerVision
Nama: Riski Yosepha Sihombing
NIM  : 4222201065
Kelas : Robotika B Pagi

#OCR Plat Nomor Kendaraan Indonesia dengan VLM (Visual Language Model)
Proyek ini bertujuan untuk melakukan Optical Character Recognition (OCR) pada dataset plat nomor kendaraan Indonesia menggunakan model VLM (seperti LLaVA) yang dijalankan melalui LM Studio secara lokal. Hasil pengenalan karakter akan dibandingkan dengan ground truth menggunakan metrik **Character Error Rate (CER)**.

Struktur Proyek:
project-root/

├── test/ # Folder berisi gambar .jpg dan label .txt

├── labels/ # Folder output label ground truth

│ └── ground_truth.csv

├── generate.py # Membuat CSV ground truth

└── ocr_eval.py # Mengirim gambar ke model dan menghitung CER

#Tahap Pengerjaan
1. 📁 Persiapan Dataset
- Dataset yang digunakan: [Indonesian License Plate Recognition](https://www.kaggle.com/datasets/juanthomaswijaya/indonesianlicense-plate-dataset)
- Folder yang digunakan: `test/` (berisi file `.jpg` dan `.txt`)
- File `.txt` berisi hasil deteksi karakter: `class_id x_center y_center width height`
  
2. 🧾 Generate Ground Truth
Generate file ground truth:
   python generate.py

- Script `generate.py` digunakan untuk membaca file `.txt` dan menyusun urutan karakter berdasarkan posisi `x_center`.
- Label numerik dikonversi menjadi karakter menggunakan `label_map`.
- Hasil disimpan ke file CSV `ground_truth.csv` dengan format:
  contoh:
image, ground_truth
ABC123.jpg, B123CDE
test001_1.jpg,B9140BCD

3.   🤖 Menjalankan Model VLM via LM Studio

Pastikan LM Studio sudah berjalan dan model ready.
   
- LM Studio diinstal secara lokal dan dijalankan di port default `http://localhost:1234`.
- Model yang digunakan: `llava-v1.5-7b` atau model VLM lain yang kompatibel dengan LM Studio.
- Pastikan model dapat menerima input gambar dan prompt melalui API lokal.
- Aktifkan GPU atau opsi offload jika tersedia—ini akan mempercepat inferensi.
  
4. 🧠 Inferensi OCR
Jalankan evaluasi OCR:
   python ocr_eval.py


- Baca `ground_truth.csv` dan file gambar dari folder `test/`.
- Untuk setiap gambar:
a. Convert gambar ke base64 atau upload melalui `image_url` JSON payload.
b. Kirim request ke endpoint `/v1/chat/completions` dengan prompt:
   ```
   "What is the license plate number shown in this image? Respond with just the plate."
   ```
c. Terima hasil OCR (prediksi) dari model.
- Simpan hasil antara prediksi dan ground truth ke CSV `results.csv`, termasuk nilai CER per gambar.
 
  5. 📊 Evaluasi: Perhitungan Character Error Rate (CER)

Cek ground_truth.csv dan results.csv setelah selesai proses.
     
- Hitung **Character Error Rate (CER)** per baris:
Menggunakan library `jiwer` atau implementasi manual untuk menghitung CER:
```python
CER = Levenshtein Distance / Jumlah karakter ground truth

Contoh Output:
image,predicted,ground_truth,CER
AB123CD.jpg,AB123CD,AB123CD,0.0
B1234XY.jpg,B123XY,B1234XY,0.14
test001_1.jpg,B91411,B9140BCD,0.625
test001_2.jpg,[ERROR],B2407UZO,1.0
...
