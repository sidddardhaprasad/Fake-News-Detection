import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle

# Load data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
fake['label'] = 0
real['label'] = 1
df = pd.concat([fake, real])

# Clean text
stop_words = stopwords.words('english')
def clean_text(text):
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['text_clean'] = df['text'].apply(clean_text)

# Split data
X = df['text_clean']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tf, y_train)

# Evaluate
y_pred = model.predict(X_test_tf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
