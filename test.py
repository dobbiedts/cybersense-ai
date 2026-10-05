from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

vectorizer = joblib.load("models/sqlvectorizer (1).pkl")
print("Fitted:", hasattr(vectorizer, "idf_"))
print("Vocabulary size:", len(vectorizer.vocabulary_))
