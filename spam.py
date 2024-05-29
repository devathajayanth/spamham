from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load SMS spam dataset
df = pd.read_csv('C:/Users/Dell/Downloads/spam.csv', encoding='latin-1')

# Text preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    # Remove phone numbers
    text = re.sub(r'\b\d{10}\b', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['text'] = df['text'].apply(preprocess_text)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy, classification_report=classification_rep)

@app.route('/classify', methods=['POST'])
def classify():
    sms_text = request.form['sms_text']
    text_clean = preprocess_text(sms_text)
    text_tfidf = vectorizer.transform([text_clean])
    prediction = clf.predict(text_tfidf)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
