import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from transformers import pipeline, logging

logging.set_verbosity_error()


@st.cache_resource
def load_embedding_model():
   
    return pipeline(
        "feature-extraction",
        model="BAAI/bge-small-en-v1.5",
        device=-1,  # CPU
    )


@st.cache_data(show_spinner="Building movie embeddings (one time only)...")
def compute_embeddings(df):

    model = load_embedding_model()

    output = df["Description"].fillna("").apply(
        lambda x: np.array(model(x, truncation=True)[0]).mean(axis=0)
    )

    return output


def emb_pipeline(selected_movie, df, recommendation_count):
    """Return the titles of the movies whose descriptions are most similar."""
    df = df.reset_index(drop=True)

    embeddings = compute_embeddings(df)

    # Turn the Series of vectors into one matrix: (number of movies, 384)
    all_embeddings = np.vstack(embeddings.values)

    matches = df[df["Title"] == selected_movie].index
    selected_embedding = all_embeddings[matches[0]].reshape(1, -1)

    similarities = cosine_similarity(selected_embedding, all_embeddings)[0]

    similarity_df = pd.DataFrame({
        "Title": df["Title"],
        "Similarities": similarities
    })

    similarity_df = similarity_df[similarity_df["Title"] != selected_movie]

    sort_df = similarity_df.sort_values(by="Similarities", ascending=False)

    return sort_df["Title"].head(recommendation_count).tolist()