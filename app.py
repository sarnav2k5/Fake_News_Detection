
import streamlit as st
import joblib
import re
import string

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(
        r"[%s]" % re.escape(string.punctuation), "", text
    )
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("📰 Fake News Detection System")
st.write("Analyze a news headline and article using machine learning.")

try:
    model, vectorizer = load_model()

    title = st.text_input("News Headline")
    article = st.text_area("News Article", height=200)

    if st.button("Analyze News"):
        content = (title + " " + article).strip()

        if not content:
            st.warning("Please enter a headline or article.")
        else:
            cleaned = clean_text(content)
            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)[0]

            probabilities = model.predict_proba(features)[0]
            classes = list(model.classes_)
            confidence = probabilities[classes.index(prediction)] * 100

            if prediction == 0:
                st.error("Prediction: Fake News")
            else:
                st.success("Prediction: Real News")

            st.metric("Model Confidence", f"{confidence:.2f}%")
            st.caption(
                "This prediction is not independent verification "
                "of the article's factual accuracy."
            )

except FileNotFoundError:
    st.error(
        "Model files are missing. Run train_model.py first "
        "to generate both .pkl files."
    )
