
An AI-powered movie recommendation engine that combines machine learning clustering with intelligent data analytics to deliver personalized movie suggestions.
Live Demo • Report Bug • Request Feature
</div>


# Overview

Movie Magic AI is a sophisticated movie recommendation system that leverages machine learning and data science to provide personalized movie suggestions. Unlike traditional recommendation systems that rely solely on genre matching, Movie Magic AI employs a hybrid approach combining:

K-Means Clustering for discovering hidden patterns in movie data
Genre-based Filtering for traditional content-based recommendations
Interactive Data Exploration for understanding movie trends and distributions
Real-time Analytics for data health monitoring and insights

The application is built with a clean, modern UI using Streamlit and features glassmorphism design elements, animated components, and an intuitive user experience.

# Key Features

1. Dual Recommendation Engines

Normal Search: Traditional genre-based matching for straightforward recommendations
Hybrid Intelligence Engine (HIE): Advanced ML clustering that discovers "hidden gem" matches across similar movies, even outside standard genre boundaries

2. Dynamic Data Exploration

Interactive Movie Analysis: Select any movie to explore its ecosystem within the dataset
Genre Statistics: Compare individual movies against genre averages
Visual Insights: Distribution plots showing how a movie compares to similar titles
Data Health Monitoring: Built-in checks for null values and data quality

3. Modern User Interface

Animated landing page with cinematic elements
Glassmorphism design cards for enhanced visual appeal
Responsive metrics dashboard with real-time comparisons
Color-coded gradients and smooth transitions

4. Machine Learning Pipeline

Automated data preprocessing and feature engineering
Frequency encoding for categorical variables (Directors, Actors)
One-hot encoding for genre classification
Standardized scaling with scikit-learn pipelines
K-Means clustering with 13 optimal clusters

# Detailed Movie Metadata

Comprehensive movie information (Genre, Director, Year, Runtime, Rating, Votes)
Plot descriptions and contextual details
Expandable sections for deeper exploration


<img width="696" height="681" alt="image" src="https://github.com/user-attachments/assets/6c333fd3-1c94-4d14-bf62-d1dd52449a77" />


# Installation

Prerequisites

Python 3.8 or higher
pip package manager
Git (optional, for cloning)

Step 1: Clone the Repository
bashgit clone https://github.com/yourusername/movie-magic-ai.git
cd movie-magic-ai
Step 2: Create Virtual Environment (Recommended)
bash# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux

python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
bashpip install -r requirements.txt
requirements.txt:
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
seaborn>=0.12.0
matplotlib>=3.7.0
Step 4: Prepare Dataset
Place your IMDb movie dataset CSV file in the data/ directory:
bashmkdir data
# Copy your imdb_movie_dataset.csv to data/
Required CSV Columns:

Title: Movie title
Genre: Movie genre
Director: Director name
Actors: Main actors (comma-separated)
Year: Release year
Rating: IMDb rating (0-10)
Votes: Number of votes
Runtime (Minutes): Movie duration
Description: Plot summary


# Usage

Running the Application Locally
bashstreamlit run app.py
The application will open in your default browser at http://localhost:8501
Deployment on Hugging Face Spaces

Create a new Space on Hugging Face
Select Streamlit as your SDK
Upload files:

app.py (main application file)
requirements.txt
data/imdb_movie_dataset.csv

<img width="717" height="480" alt="image" src="https://github.com/user-attachments/assets/302c322a-2c2f-4939-bee8-827e71462650" />
