# 📧 Email Spam Detection System

A Machine Learning project that classifies emails as **Spam** or **Ham** using text-based features and deploys the model through a **Streamlit web application**.

---

## 🔹 Overview
Spam emails pose security and productivity challenges.  
This project uses **TF-IDF vectorization** and a **Linear SVM classifier** to accurately detect spam emails from raw text.

---

## 📊 Dataset
- Real-world email dataset  
- 83,448 email messages  
- Columns:
  - `text` – email content  
  - `label` – 0 (Ham), 1 (Spam)

---

## ⚙️ Approach
- Text cleaning & normalization  
- TF-IDF feature extraction (unigrams & bigrams)  
- Linear SVM with probability calibration  
- Model evaluation using standard classification metrics  

---

## 📈 Results
- **Accuracy:** ~98%  
- Strong precision and recall for both Spam and Ham  

---

## 🚀 Deployment
- Interactive **Streamlit** web app  
- Real-time spam prediction with confidence scores  

---

## 🛠️ Tech Stack
Python | Pandas | Scikit-learn | TF-IDF | Streamlit

---
