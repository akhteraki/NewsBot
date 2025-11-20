import streamlit as st
import feedparser
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------------------------------------
# Load Models
# -------------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

embedder = load_embedder()
tokenizer, generator = load_summarizer()

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.markdown(
    """
    # Real-Time News Bot  
    Built using Transformer Models  
    **Developed by Dr. Akhter M.**
    """
)

st.write(
    """
    Enter any topic or keyword to see the latest real-time news.  
    This bot retrieves fresh headlines, ranks them using embeddings,
    and generates a clean summary using a transformer summarizer.
    """
)

query = st.text_input("Search for news related to:")

# -------------------------------------------------------------
# News Processing
# -------------------------------------------------------------
if st.button("Get Latest News"):
    if not query.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Fetching and processing news..."):
            url = "https://news.google.com/rss/search?q=" + query.replace(" ", "%20")
            feed = feedparser.parse(url)

            if len(feed.entries) == 0:
                st.error("No matching news found.")
            else:
                # Extract Titles
                titles = [entry.title for entry in feed.entries]

                # Rank using Embeddings
                embeddings = embedder.encode(titles)
                best_index = max(range(len(titles)), key=lambda i: embeddings[i].sum())

                top_title = titles[best_index]
                description = feed.entries[best_index].summary

                # Summarize
                inputs = tokenizer(description, return_tensors="pt", max_length=512, truncation=True)
                summary_ids = generator.generate(inputs["input_ids"], max_length=120)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                # Display Results
                st.markdown("## Top News Article")
                st.write(top_title)

                st.markdown("## Summary")
                st.write(summary)

                st.markdown("---")
                st.caption("Transformer-powered real-time news bot by Dr. Akhter M.")
