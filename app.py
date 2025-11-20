import streamlit as st
import feedparser
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load Models
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
generator = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

st.title("📰 Real-Time News Bot (Transformer Powered)")
query = st.text_input("Enter topic (e.g., Baramulla, AI, Elections, Sports)")

if st.button("Get News"):
    with st.spinner("Fetching news…"):
        feed = feedparser.parse("https://news.google.com/rss/search?q=" + query)

        if len(feed.entries) == 0:
            st.error("No news found.")
        else:
            titles = [entry.title for entry in feed.entries]

            # embeddings
            embeddings = embedder.encode(titles)
            best = max(range(len(titles)), key=lambda i: embeddings[i].sum())

            best_title = titles[best]
            desc = feed.entries[best].summary

            # summarization
            inputs = tokenizer(desc, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = generator.generate(inputs["input_ids"], max_length=100)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            st.subheader("Top News")
            st.write(best_title)

            st.subheader("Summary")
            st.write(summary)
