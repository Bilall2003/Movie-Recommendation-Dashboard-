from transformers import pipeline,logging
import torch
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


logging.set_verbosity_error()
@st.cache_resource

def emb_pipeline(selected_movie,all_movies,df,recommendation_count):
    
    torch.set_num_threads(4)
    
    emb_model=pipeline(
        "feature-extraction",
        model="BAAI/bge-small-en-v1.5",
        device=-1,
        use_fast=True
    )
    
    all_movies_emb=df["Description"].apply(lambda x : np.array(emb_model(x)[0]).mean(axis=0))
    
    selected_movie_index=df[df["Title"]==selected_movie].index
    
    selected_movie_emb=np.array(all_movies_emb[selected_movie_index]).reshape(1,-1)
    
    all_movies_emb_final=np.vstack(all_movies_emb)
    
    similarities=cosine_similarity(selected_movie_emb,all_movies_emb_final)
    
    similarities=pd.Series(similarities.flatten())
    
    return similarities.argsort()[::-1]
    

 
                 
        
    