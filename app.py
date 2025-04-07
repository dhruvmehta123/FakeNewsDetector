import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Load vectorizers
vectorizer = joblib.load('vectorizer.jb')      # TF-IDF: for LR, ANN
vectorizer2 = joblib.load('vectorizer2.jb')    # Count: for KNN, tree models, NB

# Load models
models = {
    "Logistic Regression": ("linear", joblib.load('lr_model.jb')),
    #"K-Nearest Neighbors": ("knn", joblib.load('knn_model.jb')),
    #"XGBoost": ("tree", joblib.load('xgb_improved_model.jb')),
    #"Random Forest": ("tree", joblib.load('rf_model.jb')),
    #"Decision Tree": ("tree", joblib.load('dt_improved_model.jb')),
    "Naive Bayes": ("bayes", joblib.load('nb_model.jb'))
    #"Artificial Neural Network": ("ann", load_model('ann_model.h5'))
}

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

            else:  # tree, knn, bayes
                vectorized_input = vectorizer2.transform([inputn])
                pred = model.predict(vectorized_input)[0]

            # Display result
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
