import streamlit as st
import joblib
import numpy as np

# Load vectorizers
vectorizer = joblib.load('vectorizer.jb')      # For Logistic Regression
vectorizer2 = joblib.load('vectorizer2.jb')    # For Naive Bayes

# Load models
models = {
    "Logistic Regression": ("linear", joblib.load('lr_model.jb')),
    "Naive Bayes": ("bayes", joblib.load('nb_model.jb'))
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
            if model_type == "linear":
                vectorized_input = vectorizer.transform([inputn])
            else:  # bayes
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
