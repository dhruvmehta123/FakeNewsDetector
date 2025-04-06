import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import os
import requests

# -------------------------------
# Google Drive download helper
# -------------------------------
def download_file_from_google_drive(file_id, dest_path):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)
            break

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# -------------------------------
# GDrive files (only for large ones)
# -------------------------------
gdrive_files = {
    "knn_model.jb":  "1Fh1A5BvPV6sJJJeIBdFKJu3cDAxvDig3",
    "rf_model.jb":   "10LLGlmX8hmYUAp1VeuKlUrTbvI2VtCZk",
    "ann_model.h5":  "1Q_Pl-gyR5wUm2b3uJUQZwLU_Y3GtPXZn"
}

for filename, file_id in gdrive_files.items():
    if not os.path.exists(filename):
        download_file_from_google_drive(file_id, filename)

# -------------------------------
# Load Models and Vectorizers
# -------------------------------
vectorizer = joblib.load('vectorizer.jb')      # TF-IDF: for LR, ANN
vectorizer2 = joblib.load('vectorizer2.jb')    # Count: for KNN, tree models, NB

models = {
    "Logistic Regression": ("linear", joblib.load('lr_model.jb')),
    "K-Nearest Neighbors": ("knn", joblib.load('knn_model.jb')),
    "Random Forest": ("tree", joblib.load('rf_model.jb')),
    "Naive Bayes": ("bayes", joblib.load('nb_model.jb')),
    "Artificial Neural Network": ("ann", load_model('ann_model.h5'))
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📰 Fake News Detection")
st.write("Enter the news article below:")

inputn = st.text_area("News Article")

if st.button("Check News"):
    if inputn.strip():
        st.subheader("🔍 Model-wise Predictions:")

        real_count = 0
        fake_count = 0

        for name, (model_type, model) in models.items():
            if model_type in ["linear", "ann"]:
                vectorized_input = vectorizer.transform([inputn])
                if model_type == "ann":
                    vectorized_input = vectorized_input.toarray()
                    pred = model.predict(vectorized_input)[0][0]
                    pred = 1 if pred >= 0.5 else 0
                else:
                    pred = model.predict(vectorized_input)[0]
            else:
                vectorized_input = vectorizer2.transform([inputn])
                pred = model.predict(vectorized_input)[0]

            if pred == 1:
                st.success(f"{name}: Real News ✅")
                real_count += 1
            else:
                st.error(f"{name}: Fake News ❌")
                fake_count += 1

        st.markdown("---")
        st.subheader("🧠 Overall Verdict (Majority Voting):")

        if real_count > fake_count:
            st.success(f"The news is **Most Likely Real** 🟢 ({real_count} out of {len(models)} models)")
        elif fake_count > real_count:
            st.error(f"The news is **Most Likely Fake** 🔴 ({fake_count} out of {len(models)} models)")
        else:
            st.warning("The models are evenly split. Verdict: **Inconclusive** ⚖️")
    else:
        st.warning("Please enter some text to analyze.")
