# 🌾 AgriNews AI - NLP News Classification Dashboard

AgriNews AI adalah sebuah aplikasi berbasis web yang menggunakan teknik **Natural Language Processing (NLP)** dan **Machine Learning** untuk mengklasifikasikan apakah sebuah teks/judul berita termasuk dalam kategori **Berita Pertanian (Agriculture)** atau **Bukan Berita Pertanian**.

Proyek ini dibangun menggunakan Python (Scikit-Learn, NLTK) untuk pemrosesan bahasa alami dan Flask sebagai *framework* antarmuka web (Dashboard).

---

## ✨ Fitur Utama
- **Text Preprocessing**: Menerapkan *Lowercasing*, *Noise Removal*, *Tokenization*, *Stopword Removal*, dan *Stemming* menggunakan NLTK.
- **Smart Classification**: Menggunakan algoritma **TF-IDF Vectorizer** dan **Multinomial Naive Bayes** untuk akurasi klasifikasi yang optimal.
- **Confidence Threshold Filter**: Dilengkapi logika penyaringan pintar. Jika keyakinan model di bawah 70%, sistem akan memberikan peringatan (Diragukan) untuk menghindari *False Positives*.
- **Out of Vocabulary (OOV) Handling**: Sistem tidak akan menebak asal-asalan jika kata yang diinputkan tidak dikenali sama sekali (Zero Vector).
- **Interactive Dashboard**: Antarmuka pengguna (UI) modern menggunakan Bootstrap dan fitur AJAX agar hasil prediksi muncul instan tanpa perlu memuat ulang halaman (*No Reload*).

---

## 🛠️ Teknologi yang Digunakan
* **Bahasa Pemrograman**: Python 3.x
* **Machine Learning & NLP**: `scikit-learn`, `pandas`, `nltk`, `joblib`
* **Web Framework**: Flask
* **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, FontAwesome

---

## 📂 Struktur Direktori Proyek

```text
dashboard-nlp/
│
├── templates/                 
│   └── index.html             # Tampilan Antarmuka (UI) Dashboard
│
├── app.py                     # Script utama Backend Web (Flask Server)
├── train_model.py             # Script untuk melatih dan membangun model NLP
├── final_dataset.csv          # Dataset berita yang sudah dibersihkan
├── model_pertanian.pkl        # File model ML (Ter-generate otomatis setelah training)
└── README.md                  # Dokumentasi Proyek
