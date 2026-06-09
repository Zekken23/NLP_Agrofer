import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download data NLTK
nltk.download('stopwords')

# Inisialisasi
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Fungsi cleaning yang SAMA PERSIS dengan di app.py"""
    text = str(text).lower() # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove noise
    words = text.split() # Tokenize
    # Stopword removal & Stemming
    cleaned_words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(cleaned_words)

# 1. Load data
print("Membaca dataset...")
df = pd.read_csv('final_dataset.csv')
df = df.dropna(subset=['news_title', 'target'])

# 2. Lakukan Preprocessing ke seluruh data training
print("Melakukan Text Preprocessing (Cleaning, Stopwords, Stemming)... Mohon tunggu...")
df['cleaned_text'] = df['news_title'].apply(preprocess_text)

# 3. Siapkan Fitur (X yang sudah bersih) dan Target (y)
X = df['cleaned_text']
y = df['target']

# 4. Buat Pipeline Model (TF-IDF + Naive Bayes)
model_pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)), 
    MultinomialNB()
)

# 5. Latih Model
print("Sedang melatih model...")
model_pipeline.fit(X, y)

# 6. Simpan Model Baru
joblib.dump(model_pipeline, 'model_pertanian.pkl')
print("✅ Model berhasil dilatih ulang dan disimpan sebagai 'model_pertanian.pkl'")