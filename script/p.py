import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from api import get_movie_data,get_trailer,TMDB_IMG_BASE
from advanced import emb_pipeline

# Set Page Config for a professional look
st.set_page_config(
    page_title="Movie Magic",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic styling
st.markdown("""
    <style>
    .main-header {
        font-size: 40px !important;
        font-weight: 700 !important;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 18px !important;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .stAlert {
        border-radius: 10px !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1E40AF;
    }
    </style>
""", unsafe_allow_html=True)


class EDA:
    def home(self):
        # Custom CSS for the landing page
        st.markdown("""
            <style>
            .main-title {
                font-size: 70px;
                font-weight: bold;
                text-align: left;
                background: linear-gradient(to right, #1E3A8A, #6dd5ed);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-top: 20px;
            }
            
            /* Animated Movie Reel Effect */
            @keyframes mergeBehindSync {
                0%, 100% { transform: translateX(30px); z-index: 1; opacity: 0.8; }
                50% { transform: translateX(120px); z-index: 0; opacity: 0.4; }
            }
            @keyframes mergeBehindSyncRight {
                0%, 100% { transform: translateX(-30px); z-index: 1; opacity: 0.8; }
                50% { transform: translateX(-120px); z-index: 0; opacity: 0.4; }
            }
            .animated-container {
                display: flex;
                justify-content: center;
                align-items: center;
                position: relative;
                width: 100%;
                height: 200px;
            }
            .animated-container img.side {
                width: 100px;
                position: absolute;
            }
            .animated-container img.left { animation: mergeBehindSync 4s infinite ease-in-out; }
            .animated-container img.right { animation: mergeBehindSyncRight 4s infinite ease-in-out; }
            .animated-container img.center {
                width: 150px;
                z-index: 2;
                filter: drop-shadow(0 0 15px #6dd5ed);
            }
            /* Glassmorphism Cards */
            .movie-card {
                background: linear-gradient(45deg, rgba(30, 58, 138, 0.7) 0%, rgba(109, 213, 237, 0.1) 100%);
                padding: 30px;
                border-radius: 15px;
                color: white;
                margin-bottom: 25px;
                transition: 0.3s ease;
                border: 1px solid rgba(255,255,255,0.1);
            }
            .movie-card:hover {
                transform: scale(1.02);
                border: 1px solid #6dd5ed;
                cursor: pointer;
            }
            .movie-card h2 { margin-top: 0; font-size: 2rem; color: #6dd5ed; }
            .movie-card p { font-size: 1.1rem; line-height: 1.6; opacity: 0.9; }
            
            /* Icon Animation */
            @keyframes iconMove {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            .card-icon {
                width: 50px;
                margin-bottom: 10px;
                animation: iconMove 3s infinite ease-in-out;
            }
            </style>
        """, unsafe_allow_html=True)

        # Animated Header Section
        st.markdown("""
            <div class="animated-container">
                <img class="side left" src="https://cdn-icons-png.flaticon.com/512/4221/4221419.png">
                <img class="center" src="https://cdn-icons-png.flaticon.com/512/3163/3163478.png">
                <img class="side right" src="https://cdn-icons-png.flaticon.com/512/4221/4221419.png">
            </div>
            <h1 class="main-title">Movie Magic AI</h1>
        """, unsafe_allow_html=True)
        
        # Landing Page Content Blocks
        st.markdown("""
            <div class="movie-card">
                <img class="card-icon" src="https://www.freeiconspng.com/uploads/artificial-intelligence-icon-11.jpg">
                <h2>Cinematic Intelligence</h2>
                <p>Experience the next generation of movie discovery. Our AI analyzes thousands of data points including genres, directors, and cast chemistry to find your next favorite film.</p>
            </div>
            
            <div class="movie-card">
                <img class="card-icon" src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png">
                <h2>Data-Driven Discovery</h2>
                <p>Upload your own dataset and watch as the engine automatically cleans and transform dataset into logical groupings.</p>
            </div>
            <div class="movie-card">
                <img class="card-icon" src="https://cdn-icons-png.flaticon.com/512/1491/1491468.png">
                <h2>Hybrid Recommendation Engine</h2>
                <p>Switch between standard Genre-matching or our proprietary Hybrid Engine that uses ML model to find "hidden gem" matches outside of standard categories.</p>
            </div>
        """, unsafe_allow_html=True)

        
    def eda(self):
        st.markdown('<p class="main-header">📊 Dynamic Movie Insights</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Select a movie to explore its ecosystem within our dataset.</p>', unsafe_allow_html=True)
        
        # 1. LOAD DATASET AUTOMATICALLY (Internal)
        if "df" not in st.session_state:
            try:
                # Replace 'your_data.csv' with your actual filename
                st.session_state.df = pd.read_csv("data/imdb_movie_dataset.csv")
                self.df = st.session_state.df
            except FileNotFoundError:
                st.error("Dataset not found! Please ensure your CSV is in the project folder.")
                return
        else:
            self.df = st.session_state.df

        # 2. DYNAMIC SEARCH & FILTER
        st.markdown("### 🔎 Search & Select")
        movie_list = self.df['Title'].values
        selected_movie = st.selectbox("Type or select a movie to analyze:", movie_list)

        # Filter data for dynamic context
        movie_data = self.df[self.df['Title'] == selected_movie].iloc[0]
        genre_context = self.df[self.df['Genre'] == movie_data['Genre']]

        st.markdown("---")

        # 3. DYNAMIC METRIC DASHBOARD
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rating", f"{movie_data['Rating']}/10")
        col2.metric("Genre", movie_data['Genre'])
        col3.metric("Year", int(movie_data['Year']))
        # Dynamically compare rating to genre average
        avg_genre_rating = genre_context['Rating'].mean()
        col4.metric("Genre Avg", f"{avg_genre_rating:.1f}", delta=f"{movie_data['Rating'] - avg_genre_rating:.1f}")

        # 4. DATA HEALTH (Global Perspective)
        with st.expander("🛠️ View Global Dataset Health"):
            data_options = ["🔍 Analyze Null Values", "🎯 Analyze Duplicate Rows"]
            user_choice = st.radio("Global Health Check:", data_options, horizontal=True)
            
            if user_choice == data_options[0]:
                null_count = self.df.isnull().sum().sum()
                if null_count == 0:
                    st.success("✅ Global Data is clean: 0 Null values found.")
                else:
                    st.warning(f"⚠️ {null_count} Null values detected in global dataset.")

        # 5. DYNAMIC VISUALIZATION
        st.markdown(f"### 📈 How '{selected_movie}' compares to other {movie_data['Genre']} movies")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(genre_context['Rating'], kde=True, color="#2193b0", ax=ax)
        # Add a line for the specific selected movie
        ax.axvline(movie_data['Rating'], color='green', linestyle='--', label=f'{selected_movie}({movie_data["Rating"]})')
        plt.title(f"Rating Distribution for {movie_data['Genre']} Genre")
        plt.legend()
        st.pyplot(fig)

        # 6. DYNAMIC STATS
        st.markdown("### 📊 Genre Statistics")
        st.write(f"Descriptive statistics for all movies in the **{movie_data['Genre']}** category:")
        st.dataframe(genre_context.describe().style.background_gradient(cmap="Blues"), use_container_width=True)

class predicter(EDA):
    
    def show_movie_details(self, movie_title, df_source):
        """Helper to display formatted metadata for a specific title"""
        movie_data = df_source[df_source["Title"] == movie_title].iloc[0]
        
        # Create a nice detail layout using columns
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(f"**🎭 Genre:** {movie_data.get('Genre', 'N/A')}")
            st.markdown(f"**🎬 Director:** {movie_data.get('Director', 'N/A')}")
        with d2:
            st.markdown(f"**📅 Year:** {movie_data.get('Year', 'N/A')}")
            st.markdown(f"**⏳ Runtime:** {movie_data.get('Runtime (Minutes)', 'N/A')} min")
        with d3:
            st.markdown(f"**⭐ Rating:** {movie_data.get('Rating', 'N/A')}/10")
            st.markdown(f"**⭐ Voted:** {movie_data.get('Votes', 'N/A')}")
        
        st.info(f"**📝 Description:** {movie_data.get('Description', 'No description available.')}")

    def predict(self):

        if "df" not in st.session_state:
            st.error("Dataset is not loaded.")
            return

        # ---------------------------------------------------------
        # LOAD ORIGINAL DATASET
        # ---------------------------------------------------------

        self.df = st.session_state.df.copy()

        st.markdown(
            '<p class="main-header">🍿 Movie Magic Engine</p>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<p class="sub-header">'
            'AI-driven recommendation with deep-dive metadata explorers.'
            '</p>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # ---------------------------------------------------------
        # SEARCH ENGINE SELECTION
        # ---------------------------------------------------------

        setting = st.sidebar.radio(
            "Select Search Engine Type",
            ["Normal Search", "Advanced Search"],
            help=(
                "Normal Search uses KMeans clustering. "
                "Advanced Search uses BGE embeddings and cosine similarity."
            )
        )

        st.sidebar.info(
            "Use Advanced Search for semantic recommendations."
        )

        # ---------------------------------------------------------
        # USER INPUTS
        # ---------------------------------------------------------

        col_sel, col_count = st.columns([2, 1])

        with col_sel:

            movie_list = self.df["Title"].dropna().tolist()

            selected_movie = st.selectbox(
                "Select a Movie you love 🌐",
                movie_list
            )

        with col_count:

            recommed_count = st.slider(
                "Recommendations count",
                1,
                5,
                3
            )

        # ---------------------------------------------------------
        # SELECTED MOVIE DETAILS
        # ---------------------------------------------------------

        with st.expander(
            f"✨ View Details for Selected: {selected_movie}",
            expanded=True
        ):

            self.show_movie_details(
                selected_movie,
                self.df
            )

        # ---------------------------------------------------------
        # GENERATE RECOMMENDATIONS
        # ---------------------------------------------------------

        st.markdown("---")

        if st.button(
            "Generate Recommendations 🚀",
            use_container_width=True
        ):

            # =====================================================
            # ADVANCED SEARCH
            # =====================================================

            if setting == "Advanced Search":

                with st.spinner(
                    "🧠 Generating semantic recommendations..."
                ):

                    recs = emb_pipeline(
                        selected_movie,
                        self.df,
                        recommed_count
                    )

            # =====================================================
            # NORMAL SEARCH
            # =====================================================

            else:

                # ---------------------------------------------
                # Keep a clean copy
                # ---------------------------------------------

                working_df = self.df.copy()

                # ---------------------------------------------
                # Encode Genre
                # ---------------------------------------------

                df_encoded = pd.get_dummies(
                    working_df,
                    columns=["Genre"],
                    drop_first=True,
                    dtype=int
                )

                # ---------------------------------------------
                # Frequency encode Director and Actors
                # ---------------------------------------------

                col2 = ["Director", "Actors"]

                for col in col2:

                    if col in df_encoded.columns:

                        freq_map = (
                            working_df[col]
                            .value_counts()
                            .to_dict()
                        )

                        df_encoded[col] = (
                            df_encoded[col]
                            .map(freq_map)
                        )

                # ---------------------------------------------
                # Remove columns that are not numerical
                # ---------------------------------------------

                df_encoded.drop(
                    ["Title", "Description"],
                    axis=1,
                    inplace=True,
                    errors="ignore"
                )

                # ---------------------------------------------
                # Keep only rows with valid ML data
                # ---------------------------------------------

                valid_idx = df_encoded.dropna().index

                df_encoded = (
                    df_encoded
                    .loc[valid_idx]
                    .reset_index(drop=True)
                )

                clean_df = (
                    working_df
                    .loc[valid_idx]
                    .reset_index(drop=True)
                )

                # ---------------------------------------------
                # Check selected movie still exists
                # ---------------------------------------------

                if selected_movie not in clean_df["Title"].values:

                    st.error(
                        "The selected movie contains missing data "
                        "required by the Normal Search model."
                    )

                    return

                # ---------------------------------------------
                # KMeans Model
                # ---------------------------------------------

                with st.spinner(
                    "🧠 ML Engine calibrating clusters..."
                ):

                    operation = make_pipeline(
                        StandardScaler(),
                        KMeans(
                            random_state=101,
                            n_init=10,
                            n_clusters=13
                        )
                    )

                    operation.fit(df_encoded)

                    clusters = operation.predict(
                        df_encoded
                    )

                # ---------------------------------------------
                # Find selected movie cluster
                # ---------------------------------------------

                selected_index = clean_df.index[
                    clean_df["Title"] == selected_movie
                ][0]

                selected_cluster = clusters[selected_index]

                # ---------------------------------------------
                # Movies in same cluster
                # ---------------------------------------------

                cluster_movies = clean_df[
                    clusters == selected_cluster
                ].copy()

                # ---------------------------------------------
                # Remove selected movie
                # ---------------------------------------------

                cluster_movies = cluster_movies[
                    cluster_movies["Title"] != selected_movie
                ]

                # ---------------------------------------------
                # Sort same-cluster movies by rating
                # ---------------------------------------------

                if "Rating" in cluster_movies.columns:

                    cluster_movies = cluster_movies.sort_values(
                        by="Rating",
                        ascending=False
                    )

                # ---------------------------------------------
                # Get recommendations
                # ---------------------------------------------

                recs = cluster_movies.head(
                    recommed_count
                )["Title"].tolist()

            # =====================================================
            # CHECK RESULTS
            # =====================================================

            if not recs:

                st.warning(
                    "No recommendations were found."
                )

                return

            # =====================================================
            # DISPLAY RESULTS
            # =====================================================

            st.markdown("## 🎯 Recommended Movies")

            for i, rec in enumerate(recs, 1):

                with st.container(border=True):

                    c1, c2 = st.columns([3, 1])

                    # -----------------------------------------
                    # Movie information
                    # -----------------------------------------

                    with c1:

                        st.markdown(
                            f"#### `{i}`. {rec}"
                        )

                        st.divider()

                        self.show_movie_details(
                            rec,
                            self.df
                        )

                    # -----------------------------------------
                    # TMDB information
                    # -----------------------------------------

                    with c2:

                        movie_data = get_movie_data(rec)

                        if movie_data:

                            poster_path = movie_data.get(
                                "poster_path"
                            )

                            movie_id = movie_data.get(
                                "id"
                            )

                            if poster_path:

                                st.image(
                                    TMDB_IMG_BASE + poster_path,
                                    use_container_width=True
                                )

                            else:

                                st.info(
                                    "Poster not available"
                                )

                            if movie_id:

                                trailer_url = get_trailer(
                                    movie_id
                                )

                                if trailer_url:

                                    with c1:

                                        st.video(
                                            trailer_url,
                                            width=500
                                        )

                                else:

                                    with c1:

                                        st.info(
                                            "No trailer found "
                                            "for this movie."
                                        )

                        else:

                            st.info(
                                "TMDB information unavailable, "
                                "but this movie is from your dataset."
                            )
                                    
                                    
class stream(predicter):
    
    def run_Home(self):
        self.home()
    
    def run_eda(self):
        self.eda()
        
    def run_prediction(self):
        self.predict()
        
    def app(self):
        st.sidebar.markdown("### Menu & Controls")
        st.sidebar.caption("You’re engaging with an AI-powered tool.")
        
        options = {
            "📟 About Page":self.run_Home,
            "📊 Data Exploration & Health": self.run_eda,
            "🎬 Movie Recommender Engine": self.run_prediction
        }
        
        key_select = st.sidebar.selectbox("Go to page", list(options.keys()))
        value_select = options[key_select]
        
        # Execute page function
        value_select()

        st.sidebar.markdown(
            """
            <div style='background-color:green; color:white; padding:8px 12px; border-radius:15px; font-weight:600; font-size:13px; text-align:center; border: 1px solid white; margin-top:655px;'>
               💚 Connected to Weaviate
            </div>
            """, 
            unsafe_allow_html=True
        )
# Execution
if __name__ == "__main__":
    app_runner = stream()
    app_runner.app()