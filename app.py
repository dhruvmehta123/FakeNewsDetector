import streamlit as st
import joblib
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

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
# Download required files
# -------------------------------
required_files = {
    "ann_model.h5": "1Fh1A5BvPV6sJJJeIBdFKJu3cDAxvDig3",  # ANN model
    "tfidf_vectorizer.jb": "YOUR_VECTORIZER_FILE_ID"  # Replace with your vectorizer file ID
}

for filename, file_id in required_files.items():
    if not os.path.exists(filename):
        try:
            download_file_from_google_drive(file_id, filename)
            st.success(f"Downloaded {filename}")
        except Exception as e:
            st.error(f"Failed to download {filename}: {str(e)}")
            st.stop()

# -------------------------------
# Load Model and Vectorizer with error handling
# -------------------------------
try:
    # Load the vectorizer
    vectorizer = joblib.load('tfidf_vectorizer.jb')
    
    # Verify the vectorizer is fitted
    if not hasattr(vectorizer, 'vocabulary_'):
        st.error("The vectorizer is not fitted. Please provide a properly fitted vectorizer.")
        st.stop()
        
    # Load the ANN model
    ann_model = load_model('ann_model.h5')
    
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

# -------------------------------
# Text preprocessing function
# -------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📰 Fake News Detection (ANN Model)")
st.write("Enter the news article below to check if it's genuine or fake:")

input_text = st.text_area("News Article", height=200)

if st.button("Check News"):
    if input_text.strip():
        try:
            # Preprocess the text
            processed_text = preprocess_text(input_text)
            
            # Vectorize the input text
            vectorized_input = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = ann_model.predict(vectorized_input.toarray())
            probability = prediction[0][0]
            
            # Display results
            st.subheader("🔍 Prediction Result:")
            
            if probability > 0.5:
                st.success(f"✅ Genuine News (confidence: {probability*100:.2f}%)")
                st.balloons()
            else:
                st.error(f"❌ Fake News (confidence: {(1-probability)*100:.2f}%)")
            
            # Show probability gauge
            st.subheader("📊 Confidence Level")
            st.progress(float(probability))
            st.caption(f"Model confidence: {probability*100:.2f}% genuine")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please enter some text to analyze.")

# Add some info
st.markdown("---")
st.info("""
This app uses an Artificial Neural Network (ANN) trained on thousands of news articles to detect fake news.
The model analyzes the text content and provides a probability score indicating how likely the news is to be genuine.
""")
