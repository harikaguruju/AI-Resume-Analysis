from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

class ResumeMLModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.model = LogisticRegression()

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

        # Save model
        with open("ml_resume_model.pkl", "wb") as f:
            pickle.dump((self.vectorizer, self.model), f)

    def load(self):
        if os.path.exists("ml_resume_model.pkl"):
            with open("ml_resume_model.pkl", "rb") as f:
                self.vectorizer, self.model = pickle.load(f)

    def predict_role(self, text):
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]
