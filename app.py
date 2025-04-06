import streamlit as st
import joblib
import numpy as np

# Preprocessor and vectorizer
preprocessor = joblib.load('preprocessor.jb')
vectorizer = joblib.load('vectorizer.jb')

# Load models
models = {
    "Logistic Regression": ("linear", joblib.load('lr_model.jb')),
    "K-Nearest Neighbors": ("knn", joblib.load('knn_model.jb')),
    "Random Forest": ("tree", joblib.load('rf_model.jb')),
    "Naive Bayes": ("bayes", joblib.load('nb_model.jb'))
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📰 Fake News Detection")
st.markdown("Check if a news article is **real or fake** using multiple ML models.")

news_input = st.text_area("Paste your news content below 👇", height=200)

selected_model_name = st.selectbox("Choose a model for prediction", list(models.keys()))

if st.button("Detect"):
    if news_input.strip() == "":
        st.warning("Please enter some news text!")
    else:
        # Preprocess input
        cleaned_input = preprocessor.transform([news_input])
        vectorized_input = vectorizer.transform(cleaned_input)

        # Load model
        model_type, model = models[selected_model_name]

        # Predict
        pred = model.predict(vectorized_input)[0]

        if pred == 1:
            st.error("❌ This news seems to be **Fake**.")
        else:
            st.success("✅ This news seems to be **Real**.")
