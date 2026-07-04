from collections import Counter
import csv
import os
import re

from flask import Flask, jsonify, render_template, request
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np


ARTIFACT_PATH = os.path.join('artifacts', 'agri_nlp_experiments.joblib')
BERT_MODEL_NAME = 'indobenchmark/indobert-base-p2'

app = Flask(__name__)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

ARTIFACTS = joblib.load(ARTIFACT_PATH)
MODELS = ARTIFACTS['models']
MODEL_OPTIONS = [
    {'id': model_id, 'label': model_data['scenario'], 'feature': model_data['feature_type']}
    for model_id, model_data in MODELS.items()
]
DEFAULT_MODEL_ID = ARTIFACTS.get('best_model_id') or MODEL_OPTIONS[0]['id']
_bert_runtime = None


EXAMPLES = [
    "USDA Expanding Crop Insurance Access for organic farmers",
    "Paddy transplantation falls 5 percent, output may drop this year",
    "New pesticide revealed to be hazardous for vegetable harvest",
    "Central bank cuts interest rates to boost economic growth",
    "AI startup releases upgraded model with domestic chip support",
    "Farmers honored with National Excellence in Farming Award",
    "The rise of sustainable soybeans in New York agriculture",
    "Football champions finish bottom after narrow loss to Karnali",
    "Global wheat prices surge amid supply chain disruptions",
    "Irrigation systems improved in dryland areas to support maize production",
    "Technology sector sees record investment in semi-conductor research",
    "Organic fertilizer usage increases among small-scale coffee growers"
]


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0


def percentage(value):
    return round(float(value) * 100, 1)


def read_rows():
    rows = []
    with open('final_dataset.csv', newline='', encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            title = (row.get('news_title') or '').strip()
            target = str(row.get('target') or '').strip()
            if title and target in {'0', '1'}:
                cleaned = preprocess_text(title)
                rows.append({
                    'title': title,
                    'target': target,
                    'cleaned': cleaned,
                    'word_count': len(title.split()),
                    'token_count': len(cleaned.split())
                })
    return rows


def get_primary_vectorizer():
    best_model = MODELS[DEFAULT_MODEL_ID]
    if best_model.get('vectorizer_key'):
        return ARTIFACTS['vectorizers'][best_model['vectorizer_key']]
    return ARTIFACTS['vectorizers'].get('tfidf')


def top_feature_rows(vectorizer, cleaned_docs, feature_type):
    if vectorizer is None or not hasattr(vectorizer, 'get_feature_names_out'):
        return []
    matrix = vectorizer.transform(cleaned_docs)
    mean_scores = matrix.mean(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    top_indices = mean_scores.argsort()[::-1][:15]
    max_score = float(mean_scores[top_indices[0]]) if len(top_indices) else 0
    cutoff = max_score * 0.55
    return [
        {
            'rank': index + 1,
            'feature': feature_names[feature_index],
            'score': round(float(mean_scores[feature_index]), 5),
            'relevance': 'Tinggi' if float(mean_scores[feature_index]) >= cutoff else 'Sedang',
            'feature_type': feature_type
        }
        for index, feature_index in enumerate(top_indices)
        if mean_scores[feature_index] > 0
    ]


def class_feature_rows(model_data, vectorizer):
    classifier = model_data['estimator']
    if vectorizer is None or not hasattr(classifier, 'feature_log_prob_'):
        return {'0': [], '1': []}
    feature_names = vectorizer.get_feature_names_out()
    features = {'0': [], '1': []}
    for class_index, class_value in enumerate(classifier.classes_):
        key = str(class_value)
        indices = classifier.feature_log_prob_[class_index].argsort()[::-1][:10]
        features[key] = [
            {'feature': feature_names[index], 'score': round(float(classifier.feature_log_prob_[class_index][index]), 4)}
            for index in indices
        ]
    return features


def build_metric_card(result):
    confusion = result.get('confusion', {})
    per_class = result.get('per_class', [])
    return {
        'scope': 'Notebook train-test split 80:20',
        'train_size': ARTIFACTS['split']['train_size'],
        'test_size': ARTIFACTS['split']['test_size_rows'],
        'accuracy': percentage(result['accuracy']),
        'precision': percentage(result['precision']),
        'recall': percentage(result['recall']),
        'f1': percentage(result['f1']),
        'confusion': {
            'tp': int(confusion.get('tp', 0)),
            'tn': int(confusion.get('tn', 0)),
            'fp': int(confusion.get('fp', 0)),
            'fn': int(confusion.get('fn', 0))
        },
        'per_class': [
            {
                'class_name': 'Berita Pertanian' if item['label'] == 1 else 'Bukan Pertanian',
                'precision': percentage(item['precision']),
                'recall': percentage(item['recall']),
                'f1': percentage(item['f1']),
                'support': int(item['support']),
                'status': 'Baik' if item['f1'] >= 0.8 else 'Perlu Validasi'
            }
            for item in per_class
        ]
    }


def build_dashboard_data():
    rows = read_rows()
    label_counts = Counter(row['target'] for row in rows)
    cleaned_docs = [row['cleaned'] for row in rows]
    agriculture_docs = [row['cleaned'] for row in rows if row['target'] == '1']
    token_counter = Counter()
    for doc in cleaned_docs:
        token_counter.update(doc.split())

    best_model = MODELS[DEFAULT_MODEL_ID]
    best_result = ARTIFACTS['results'][DEFAULT_MODEL_ID]
    vectorizer = get_primary_vectorizer()
    feature_type = best_model['feature_type']
    vocabulary_size = len(vectorizer.vocabulary_) if vectorizer is not None and hasattr(vectorizer, 'vocabulary_') else 0
    feature_names = list(vectorizer.get_feature_names_out()) if vectorizer is not None and hasattr(vectorizer, 'get_feature_names_out') else []
    ngram_range = getattr(vectorizer, 'ngram_range', None)
    ngram_label = f"{ngram_range[0]}-{ngram_range[1]} gram" if ngram_range else '-'
    unigram_count = sum(1 for feature in feature_names if ' ' not in feature)
    bigram_count = vocabulary_size - unigram_count
    all_matrix = vectorizer.transform(cleaned_docs) if vectorizer is not None and cleaned_docs else None
    avg_active_features = round(all_matrix.nnz / len(rows), 1) if all_matrix is not None and rows else 0
    feature_density = round((all_matrix.nnz / (all_matrix.shape[0] * all_matrix.shape[1])) * 100, 2) if all_matrix is not None and rows and vocabulary_size else 0
    word_count_total = sum(row['word_count'] for row in rows)
    token_count_total = sum(row['token_count'] for row in rows)
    avg_words = round(word_count_total / len(rows), 1) if rows else 0
    avg_tokens = round(token_count_total / len(rows), 1) if rows else 0
    removed_pct = round((1 - (token_count_total / word_count_total)) * 100, 1) if word_count_total else 0

    model_results = [
        {
            'id': model_id,
            'scenario': model_data['scenario'],
            'feature': model_data['feature_type'],
            'classifier': model_data['classifier'],
            'accuracy': percentage(result['accuracy']),
            'precision': percentage(result['precision']),
            'recall': percentage(result['recall']),
            'f1': percentage(result['f1']),
            'is_best': model_id == DEFAULT_MODEL_ID
        }
        for model_id, model_data in MODELS.items()
        for result in [ARTIFACTS['results'][model_id]]
    ]
    model_results.sort(key=lambda item: int(item['id'].split('-')[1]))
    scenario_metrics = {
        model_id: {
            'id': model_id,
            'scenario': MODELS[model_id]['scenario'],
            'feature': MODELS[model_id]['feature_type'],
            **build_metric_card(ARTIFACTS['results'][model_id])
        }
        for model_id in MODELS
    }

    distribution = [
        {'label': 'Berita Pertanian', 'count': label_counts.get('1', 0), 'pct': round(safe_divide(label_counts.get('1', 0), len(rows)) * 100, 1), 'color': '#16c784'},
        {'label': 'Bukan Pertanian', 'count': label_counts.get('0', 0), 'pct': round(safe_divide(label_counts.get('0', 0), len(rows)) * 100, 1), 'color': '#0b7cf5'}
    ]

    examples = {
        'agriculture': [row['title'] for row in rows if row['target'] == '1'][:4],
        'non_agriculture': [row['title'] for row in rows if row['target'] == '0'][:4]
    }

    return {
        'summary': {
            'total_docs': len(rows),
            'agriculture_docs': label_counts.get('1', 0),
            'non_agriculture_docs': label_counts.get('0', 0),
            'classes': len(label_counts),
            'vocabulary_size': vocabulary_size,
            'ngram_range': ngram_label,
            'avg_words': avg_words,
            'avg_tokens': avg_tokens,
            'removed_pct': removed_pct,
            'classifier': best_model['classifier'],
            'best_model': best_model['scenario'],
            'best_model_id': DEFAULT_MODEL_ID,
            'experiment_count': len(MODELS)
        },
        'distribution': distribution,
        'top_features': top_feature_rows(vectorizer, agriculture_docs or cleaned_docs, feature_type),
        'class_features': class_feature_rows(best_model, vectorizer),
        'top_tokens': [{'token': token, 'count': count} for token, count in token_counter.most_common(12)],
        'feature_stats': {
            'unigram_count': unigram_count,
            'bigram_count': bigram_count,
            'avg_active_features': avg_active_features,
            'feature_density': feature_density,
            'ngram_distribution': [
                {'name': 'Unigram', 'count': unigram_count, 'pct': round(safe_divide(unigram_count, vocabulary_size) * 100, 1)},
                {'name': 'Bigram', 'count': bigram_count, 'pct': round(safe_divide(bigram_count, vocabulary_size) * 100, 1)}
            ]
        },
        'classification_metrics': build_metric_card(best_result),
        'model_results': model_results,
        'scenario_metrics': scenario_metrics,
        'preprocessing_steps': [
            {'name': 'Lowercase', 'before': 'Organic Fertilizer Usage', 'after': 'organic fertilizer usage'},
            {'name': 'Noise Removal', 'before': 'rice output may drop 7%', 'after': 'rice output may drop'},
            {'name': 'Stopword Removal', 'before': 'farmers in the field', 'after': 'farmers field'},
            {'name': 'Porter Stemming', 'before': 'farmers farming crops', 'after': 'farmer farm crop'}
        ],
        'processed_samples': [
            {'title': row['title'], 'cleaned': row['cleaned'], 'target': 'Pertanian' if row['target'] == '1' else 'Non Pertanian'}
            for row in rows[:8]
        ],
        'examples': examples,
        'pipeline': [
            {'step': 'Data Cleaning', 'desc': 'Drop missing and duplicate titles from final_dataset.csv', 'state': 'done'},
            {'step': 'Preprocessing', 'desc': 'Lowercase, alphabet filter, stopword removal, Porter stemming', 'state': 'done'},
            {'step': 'Feature Extraction', 'desc': 'BoW, N-Gram, TF-IDF, Word2Vec, and IndoBERT embeddings', 'state': 'done'},
            {'step': 'Modeling', 'desc': 'Decision Tree plus Naive Bayes variants across 10 notebook scenarios', 'state': 'done'},
            {'step': 'Dashboard Runtime', 'desc': 'Select saved notebook model and run live prediction', 'state': 'now'}
        ],
        'model_cards': [
            {'name': 'BoW / N-Gram / TF-IDF', 'meta': '6 scenarios', 'desc': 'Sparse vectorizers paired with Decision Tree and Multinomial Naive Bayes.'},
            {'name': 'Word2Vec', 'meta': '2 scenarios', 'desc': 'Mean document vectors from the notebook Word2Vec model.'},
            {'name': 'IndoBERT', 'meta': '2 scenarios', 'desc': 'Transformer sentence embeddings for Decision Tree and Gaussian Naive Bayes.'},
            {'name': 'Best Split Model', 'meta': DEFAULT_MODEL_ID, 'desc': best_model['scenario']}
        ]
    }


def get_w2v_vector(cleaned_text):
    model = ARTIFACTS['word2vec']
    tokens = word_tokenize(cleaned_text.lower())
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros((1, ARTIFACTS['word2vec_vector_size']))
    return np.mean(vectors, axis=0).reshape(1, -1)


def get_bert_runtime():
    global _bert_runtime
    if _bert_runtime is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        _bert_runtime = {'tokenizer': tokenizer, 'model': model, 'device': device, 'torch': torch}
    return _bert_runtime


def get_bert_vector(text):
    runtime = get_bert_runtime()
    tokenizer = runtime['tokenizer']
    model = runtime['model']
    device = runtime['device']
    torch = runtime['torch']
    encoded = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors='pt')
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    last_hidden = outputs.last_hidden_state
    mask = encoded['attention_mask'].unsqueeze(-1).float()
    embedding = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return embedding.cpu().numpy()


def vectorize_for_model(model_data, raw_text, cleaned_text):
    feature_type = model_data['feature_type']
    if feature_type in {'bow', 'ngram', 'tfidf'}:
        vectorizer = ARTIFACTS['vectorizers'][model_data['vectorizer_key']]
        text_vector = vectorizer.transform([cleaned_text])
        return text_vector, text_vector.nnz
    if feature_type == 'word2vec':
        text_vector = get_w2v_vector(cleaned_text)
        return text_vector, int(np.count_nonzero(text_vector))
    if feature_type == 'bert':
        text_vector = get_bert_vector(raw_text)
        return text_vector, int(np.count_nonzero(text_vector))
    raise ValueError(f"Unsupported feature type: {feature_type}")


DASHBOARD_DATA = build_dashboard_data()


@app.route('/')
def index():
    return render_template('index.html', examples=EXAMPLES, dashboard=DASHBOARD_DATA, model_options=MODEL_OPTIONS)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    news_text = data.get('news_text', '')
    model_id = data.get('model_id') or DEFAULT_MODEL_ID

    if not news_text:
        return jsonify({'error': 'Teks kosong'}), 400
    if model_id not in MODELS:
        return jsonify({'error': f'Model tidak dikenal: {model_id}'}), 400

    model_data = MODELS[model_id]
    cleaned_text = preprocess_text(news_text)
    text_vector, active_features = vectorize_for_model(model_data, news_text, cleaned_text)

    if model_data['feature_type'] in {'bow', 'ngram', 'tfidf'} and active_features == 0:
        return jsonify({
            'result': 'Tidak Dikenali',
            'status': 'secondary',
            'confidence': 0,
            'model': model_data['scenario'],
            'cleaned_text': cleaned_text,
            'desc': 'Teks tidak mengandung kata kunci yang dikenali oleh vectorizer notebook.'
        })

    classifier = model_data['estimator']
    pred_class = int(classifier.predict(text_vector)[0])
    if hasattr(classifier, 'predict_proba'):
        confidence = float(np.max(classifier.predict_proba(text_vector)[0]) * 100)
    else:
        confidence = 100.0

    is_agriculture = pred_class == 1
    return jsonify({
        'result': 'Berita Pertanian' if is_agriculture else 'Bukan Berita Pertanian',
        'status': 'success' if is_agriculture else 'danger',
        'confidence': round(confidence, 2),
        'model': model_data['scenario'],
        'cleaned_text': cleaned_text,
        'desc': 'Prediksi dijalankan memakai model tersimpan dari notebook.'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
