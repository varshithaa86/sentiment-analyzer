
import streamlit as st
from train_model import model

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f0ff 0%, #fff4f8 50%, #eef6ff 100%);
}
.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #6d28d9, #a855f7, #ec4899);
    padding: 38px 30px;
    border-radius: 24px;
    color: white;
    text-align: center;
    box-shadow: 0 12px 30px rgba(109, 40, 217, 0.20);
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 8px;
}
.hero p {
    color: #f5eaff;
    font-size: 17px;
}
.section-title {
    color: #54219a;
    font-size: 25px;
    font-weight: 700;
    margin-top: 22px;
}
.info-card {
    background: rgba(255,255,255,0.90);
    border: 1px solid #eadcff;
    padding: 22px;
    border-radius: 18px;
    min-height: 145px;
    box-shadow: 0 5px 18px rgba(80,40,120,0.06);
}
.info-card h3 {
    color: #6d28d9;
    margin-top: 0;
}
.info-card p {
    color: #554b66;
}
.stTextArea textarea {
    background: white !important;
    border: 1px solid #d8b4fe !important;
    border-radius: 14px !important;
    color: #292039 !important;
}
.stButton > button {
    border-radius: 12px;
    border: 1px solid #d8b4fe;
    font-weight: 600;
    transition: 0.2s;
}
.stButton > button:hover {
    border-color: #9333ea;
    color: #7e22ce;
}
.footer {
    text-align: center;
    color: #78658c;
    padding-top: 28px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero">
    <h1>💜 Sentiment Analyzer</h1>
    <p>Discover the emotion behind every review using Machine Learning.</p>
    <p>✨ Smart Text Analysis &nbsp; • &nbsp; 🤖 AI-powered Classification</p>
</div>
""", unsafe_allow_html=True)

# Information cards
st.markdown('<div class="section-title">🌟 Explore the analyzer</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>😊 Positive</h3>
        <p>Identify reviews that express satisfaction, happiness or appreciation.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>😞 Negative</h3>
        <p>Identify reviews that express dissatisfaction or disappointment.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 ML Model</h3>
        <p>Uses TF-IDF text features and Logistic Regression classification.</p>
    </div>
    """, unsafe_allow_html=True)

# Input section
st.markdown('<div class="section-title">✍️ Analyze your review</div>',
            unsafe_allow_html=True)

st.write("Enter a product review below or try one of the sample reviews.")

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

sample1, sample2 = st.columns(2)

with sample1:
    if st.button("😊 Try positive sample", use_container_width=True):
        st.session_state.review_text = (
            "This product is amazing and the quality is excellent"
        )

with sample2:
    if st.button("😞 Try negative sample", use_container_width=True):
        st.session_state.review_text = (
            "This product is terrible and the quality is very poor"
        )

review = st.text_area(
    "Your review",
    key="review_text",
    height=130,
    placeholder="Example: I love this product. It works perfectly!"
)

if st.button("✨ Analyze Sentiment", type="primary",
             use_container_width=True):

    if review.strip():
        prediction = model.predict([review.strip()])[0]

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Analysis Result</div>',
                    unsafe_allow_html=True)

        if prediction == "Positive":
            st.success("😊 Positive Sentiment")
            st.markdown("""
            <div style="background:#dcfce7; padding:20px;
                        border-radius:16px; border:1px solid #86efac;">
                <h3 style="color:#166534;">🌈 A positive opinion!</h3>
                <p style="color:#166534;">
                    The model classified this review as Positive.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("😞 Negative Sentiment")
            st.markdown("""
            <div style="background:#ffe4e6; padding:20px;
                        border-radius:16px; border:1px solid #fda4af;">
                <h3 style="color:#9f1239;">💭 A negative opinion</h3>
                <p style="color:#9f1239;">
                    The model classified this review as Negative.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            "Note: Predictions depend on the training dataset. "
            "This is an educational demonstration, not a perfect classifier."
        )
    else:
        st.warning("Please enter a review before analyzing.")

# Project information
st.markdown("---")
st.markdown('<div class="section-title">⚙️ About this project</div>',
            unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown("""
    **🛠️ Technologies used**
    - Python
    - Scikit-learn
    - TF-IDF Vectorizer
    - Logistic Regression
    - Streamlit
    """)

with right:
    st.markdown("""
    **🎯 Project objective**
    - Process text reviews
    - Convert text into numerical features
    - Predict positive or negative sentiment
    - Demonstrate a machine learning application
    """)

st.markdown("""
<div class="footer">
    MACHINE LEARNING LAB • EXPERIMENT 11<br>
    Sentiment Analyzer | Student Mini Project 💜
</div>
""", unsafe_allow_html=True)
