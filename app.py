import streamlit as st
import joblib
import re

# Load the trained ML model
model = joblib.load("sentiment_model.pkl")

# Positive words and phrases
positive_words = [
    "love", "like", "amazing", "good", "excellent",
    "great", "wonderful", "happy", "satisfied",
    "fantastic", "recommend", "perfect", "enjoy",
    "awesome", "best"
]

# Negative words and phrases
negative_words = [
    "hate", "terrible", "bad", "awful", "poor",
    "worst", "disappoint", "useless", "regret",
    "waste", "horrible"
]

def analyze_sentiment(review):

    text = review.lower().strip()

    # Check negative phrases first
    negative_phrases = [
        "don't like", "dont like", "do not like",
        "does not like", "not good", "not happy",
        "not satisfied", "not worth",
        "don't recommend", "do not recommend"
    ]

    for phrase in negative_phrases:
        if phrase in text:
            return "Negative"

    # Check negative words
    for word in negative_words:
        if re.search(r"\b" + re.escape(word) + r"\w*\b", text):
            return "Negative"

    # Check positive words
    for word in positive_words:
        if re.search(r"\b" + re.escape(word) + r"\w*\b", text):
            return "Positive"

    # Use ML model for other sentences
    return model.predict([review])[0]


# Streamlit application
st.title("Sentiment Analyzer")

st.write("Enter a review to check whether it is Positive or Negative.")

user_review = st.text_input("Enter your review:")

if st.button("Analyze"):

    if user_review.strip():

        result = analyze_sentiment(user_review)

        st.subheader("Result:")

        if result == "Positive":
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😞")

    else:
        st.warning("Please enter a review.")