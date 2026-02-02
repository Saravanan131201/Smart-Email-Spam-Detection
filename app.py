import streamlit as st
import numpy as np
import re
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("spam_classifier.pkl")
    tfidf  = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

# Load once and reuse
model, tfidf = load_model()

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'http\S+|www.\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()

# Streamlit UI
st.title("📧 Smart Email Spam Detection")

sample_text = st.text_area("Enter your message here:", height=300)

if st.button("Predict"):    
    if sample_text.strip() == "":
        st.warning("Please enter a message to predict.")
    else:
        # Clean
        cleaned = clean_text(sample_text)
        vector = tfidf.transform([cleaned])

        # Predict class
        prediction = model.predict(vector)[0]
        prob_spam = model.predict_proba(vector)[0][1]
        prob_ham  = model.predict_proba(vector)[0][0]

        if prediction == 1: 
            st.error("Prediction: Spam")
            st.progress(prob_spam)
            st.write(f"Confidence: {round(prob_spam*100,2)}%")
        else:  
            st.success("Prediction: Ham")
            st.progress(prob_ham)
            st.write(f"Confidence: {round(prob_ham*100,2)}%")