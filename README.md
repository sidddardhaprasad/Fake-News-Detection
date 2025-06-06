# Fake-News-Detection
# 📰 Fake News Detection using Machine Learning

This project is an AI-based system designed to classify news articles as **Real** or **Fake** using machine learning and natural language processing (NLP). The system can analyze manually inputted text or extract articles from URLs using the `newspaper3k` library.

## 📌 Features

- Classifies news articles as Real or Fake
- Supports text input or live URL scraping
- Streamlit web interface for easy use
- Model built with Logistic Regression and TF-IDF

## 📂 Dataset

- Source: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Files used:
  - `Fake.csv` (fake news articles)
  - `True.csv` (real news articles)

## ⚙️ Tech Stack

- Python 3.x
- pandas, scikit-learn, nltk, streamlit
- newspaper3k (for scraping article content)
- VS Code (IDE)

## 🧠 ML Approach

- **Text Cleaning**: Lowercasing, stopword and punctuation removal
- **Feature Extraction**: TF-IDF Vectorizer
- **Model**: Logistic Regression
- **Accuracy**: Evaluated using classification report and accuracy score

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Fake-News-Detection.git
   cd Fake-News-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model training:
   ```bash
   python main.py
   ```

4. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🧪 Output

Below is a sample prediction screen:

![Prediction Screenshot] ![false](https://github.com/user-attachments/assets/d7709d52-c0b4-4e5c-9d1d-8f3b0ef0489e)
                         ![true](https://github.com/user-attachments/assets/ee721843-baf2-4b6b-a051-e4a7b41f53c6)


## 🔮 Future Scope

- Add deep learning models (LSTM, BERT)
- Multilingual news support
- Real-time news dashboard
- Explainable AI (SHAP, LIME)

## 📄 License

This project is licensed under the MIT License.

## 🔗 References

- Kaggle Dataset
- newspaper3k Documentation
- scikit-learn and Streamlit Docs
