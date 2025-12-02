import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import hstack

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Clickbait word list
# -----------------------------
clickbait_words = [
    "shocking","amazing","unbelievable","secret","revealed","exposed","unexpected",
    "surprising","truth","hidden","mystery","crazy","insane","worst","best","terrifying",
    "heartbreaking","beautiful","incredible","epic","dangerous","wild",
    "dead","dies","killed","death","kills","shocked","disaster","fight","attack","accident",
    "biggest","richest","ultimate","exclusive","ridiculous",
    "21","17","10","15","20","30","50","99","5","7","3",
    "reasons","things","ways","facts","tips","hacks","signs","list",
    "you","your","people","everyone","must see","dont miss","read this",
    "viral","trending","breaking","scandal","gossip","celebrity"
]

# Topic label names for LDA topics
topic_labels = {
    0: "Politics / Government",
    1: "Business / Economy",
    2: "Technology / Gadgets / AI",
    3: "Sports / Games",
    4: "Health / Fitness / Medicine",
    5: "Science / Space / Environment",
    6: "Entertainment / Celebrity / Movies",
    7: "Crime / Accident / Shock",
    8: "Lifestyle / Travel / Fashion",
    9: "List / Tips / Hacks / Viral"
}

# -----------------------------
# Helper functions
# -----------------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

def count_clickbait_words(text: str) -> int:
    return sum(1 for w in clickbait_words if w in text)

@st.cache_resource
def train_models():
    # Load dataset (must be in same folder as app.py)
    df = pd.read_csv("clickbait_data.csv")

    # Preprocess
    df["clean_headline"] = df["headline"].apply(preprocess)
    df["cb_word_count"] = df["clean_headline"].apply(count_clickbait_words)

    # TF-IDF
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(df["clean_headline"])

    # Add engineered feature
    X_extra = df["cb_word_count"].values.reshape(-1, 1)
    X_final = hstack([X_tfidf, X_extra])

    y = df["clickbait"]

    # Logistic Regression model
    model = LogisticRegression(max_iter=300)
    model.fit(X_final, y)

    # LDA Topic model (only on TF-IDF)
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    lda.fit(X_tfidf)

    return model, tfidf, lda

def predict_lda_topic(headline: str, tfidf, lda) -> int:
    clean = preprocess(headline)
    vec = tfidf.transform([clean])
    topic_dist = lda.transform(vec)
    return int(topic_dist.argmax())

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Clickbait Detector", page_icon="📰")

st.title("Clickbait Headline Detection App 🚀")
st.write(
    "This app uses NLP, Machine Learning and Topic Modelling (LDA) "
    "to detect whether a news headline is **clickbait** and to guess its topic category."
)

with st.spinner("Training / loading model (only first time)..."):
    model, tfidf, lda = train_models()

headline = st.text_input("Enter a news headline:")

if st.button("Predict"):
    if not headline.strip():
        st.warning("Please enter a headline.")
    else:
        clean = preprocess(headline)
        cb_count = count_clickbait_words(clean)
        vec = tfidf.transform([clean])
        final_input = hstack([vec, np.array(cb_count).reshape(1, -1)])

        pred = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0][1] * 100

        topic_idx = predict_lda_topic(headline, tfidf, lda)
        topic_name = topic_labels.get(topic_idx, f"Topic {topic_idx}")

        st.write("---")
        st.write(f"**Clickbait trigger words detected:** {cb_count}")

        if pred == 1:
            st.success(f"Result: **CLICKBAIT 😱** — Probability: {prob:.2f}%")
        else:
            st.info(f"Result: **NOT CLICKBAIT 🙂** — Probability: {prob:.2f}%")

        st.write(f"**Topic Category:** {topic_name}")
        st.write("---")
        st.caption("Model: Logistic Regression + TF-IDF + Clickbait word features + LDA topic modelling.")
