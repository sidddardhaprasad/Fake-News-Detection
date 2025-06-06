import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = stopwords.words('english')

def clean_text(text):
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Streamlit UI
st.title("📰 Fake News Detector")

input_text = st.text_area("Enter News Article Text Here:")

if st.button("Check"):
    cleaned = clean_text(input_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        st.success("✅ This looks like **Real News**")
    else:
        st.error("❌ This looks like **Fake News**")
