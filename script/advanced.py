from transformers import pipeline, logging
import torch
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


logging.set_verbosity_error()


@st.cache_resource
def load_embedding_model():

    torch.set_num_threads(4)

    return pipeline(
        "feature-extraction",
        model="BAAI/bge-small-en-v1.5",
        device=-1,
        use_fast=True
    )


def emb_pipeline(selected_movie, df, recommendation_count):

    # Load cached model
    emb_model = load_embedding_model()

    # Reset index so embedding positions and dataframe positions match
    df = df.reset_index(drop=True)

    # Find selected movie
    selected_index = df.index[
        df["Title"] == selected_movie
    ].tolist()

    if not selected_index:
        return []

    selected_index = selected_index[0]

    # Create embeddings for all descriptions
    all_movies_emb = df["Description"].fillna("").apply(
        lambda x: np.array(
            emb_model(x)[0]
        ).mean(axis=0)
    )

    # Convert Series of vectors into matrix
    all_movies_emb_final = np.vstack(
        all_movies_emb.values
    )

    # Selected movie embedding
    selected_movie_emb = all_movies_emb_final[
        selected_index
    ].reshape(1, -1)

    # Calculate cosine similarity
    similarities = cosine_similarity(
        selected_movie_emb,
        all_movies_emb_final
    )[0]

    # Create similarity dataframe
    similarity_df = pd.DataFrame({
        "Title": df["Title"],
        "Similarity": similarities
    })

    # Remove selected movie itself
    similarity_df = similarity_df[
        similarity_df["Title"] != selected_movie
    ]

    # Sort by similarity
    similarity_df = similarity_df.sort_values(
        "Similarity",
        ascending=False
    )

    # Return movie titles
    return similarity_df.head(
        recommendation_count
    )["Title"].tolist()