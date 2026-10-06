import joblib
import streamlit as st
from pathlib import Path

# Get the project folder path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"

# Page configuration
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detector")
st.write("Enter an email or message to check whether it is spam.")

# Check whether the trained model exists
if not MODEL_PATH.exists():
    st.error("Model not found. Please train the model first.")
    st.info("Run: python src/train.py")
    st.stop()

# Load trained model
model = joblib.load(MODEL_PATH)

# Message input
message = st.text_area(
    "Enter your message:",
    height=180,
    placeholder="Type or paste your email/message here..."
)

# Analyze button
if st.button("🔍 Analyze Message"):

    message = message.strip()

    if not message:
        st.warning("Please enter a message.")

    elif len(message) < 3:
        st.warning("Message is too short. Please enter a longer message.")

    else:
        prediction = model.predict([message])[0]
        probability = model.predict_proba([message])[0]

        spam_probability = probability[1] * 100
        ham_probability = probability[0] * 100

        if prediction == 1:
            st.error("🚨 SPAM")
            confidence = spam_probability
        else:
            st.success("✅ NOT SPAM")
            confidence = ham_probability

        st.metric("Confidence", f"{confidence:.2f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Spam Probability", f"{spam_probability:.2f}%")

        with col2:
            st.metric("Not Spam Probability", f"{ham_probability:.2f}%")

        st.progress(int(confidence))

st.divider()

st.caption("Model: Multinomial Naive Bayes + TF-IDF")