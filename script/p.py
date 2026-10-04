import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from api import get_movie_data, get_trailer, TMDB_IMG_BASE
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


@st.cache_data(show_spinner=False)
def compute_clusters(features: pd.DataFrame, n_clusters: int = 13):
    """Scale the features and run KMeans once. Cached so it does not refit on every rerun.

    Returns the scaled feature matrix and the cluster label of every row.
    """
    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(random_state=101, n_init=10, n_clusters=n_clusters).fit_predict(scaled)
    return scaled, labels


class EDA:
    def home(self):
        # Styling and animations for the landing page
        st.markdown("""
            <style>
            /* ---------- Hero ---------- */
            .hero {
                display: flex;
                align-items: center;
                gap: 32px;
                flex-wrap: wrap;
                margin: 10px 0 35px 0;
            }
            .hero-icon svg {
                width: 150px;
                height: 150px;
                animation: heroGlow 3s ease-in-out infinite;
            }
            .hero-icon .reel {
                transform-box: fill-box;
                transform-origin: center;
                animation: spin 14s linear infinite;
            }
            .main-title {
                font-size: 64px;
                font-weight: 800;
                margin: 0;
                padding: 0;
                background: linear-gradient(to right, #1E3A8A, #6dd5ed);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .hero-sub {
                font-size: 1.2rem;
                line-height: 1.6;
                max-width: 640px;
                margin: 8px 0 16px 0;
                opacity: 0.85;
            }
            .chip {
                display: inline-block;
                padding: 6px 14px;
                margin: 0 8px 8px 0;
                border-radius: 999px;
                font-size: 0.85rem;
                font-weight: 600;
                color: #6dd5ed;
                background: rgba(30, 58, 138, 0.35);
                border: 1px solid rgba(109, 213, 237, 0.45);
            }

            /* ---------- Feature cards ---------- */
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
                gap: 24px;
                margin-bottom: 25px;
            }
            .movie-card {
                background: linear-gradient(45deg, rgba(30, 58, 138, 0.75) 0%, rgba(109, 213, 237, 0.12) 100%);
                padding: 28px;
                border-radius: 16px;
                color: white;
                transition: 0.3s ease;
                border: 1px solid rgba(255,255,255,0.12);
            }
            .movie-card:hover {
                transform: translateY(-6px) scale(1.01);
                border: 1px solid #6dd5ed;
                box-shadow: 0 10px 30px rgba(109, 213, 237, 0.25);
            }
            .movie-card h2 { margin: 14px 0 8px 0; font-size: 1.6rem; color: #6dd5ed; }
            .movie-card p { font-size: 1.02rem; line-height: 1.65; opacity: 0.92; margin: 0; }

            /* ---------- Animated icon badge ---------- */
            .icon-badge {
                position: relative;
                width: 68px;
                height: 68px;
                border-radius: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #1E3A8A, #2193b0 60%, #6dd5ed);
                box-shadow: 0 0 22px rgba(109, 213, 237, 0.5);
                animation: floatY 3.2s ease-in-out infinite;
            }
            .icon-badge::after {
                content: "";
                position: absolute;
                inset: 0;
                border-radius: 18px;
                border: 2px solid rgba(109, 213, 237, 0.7);
                animation: ringPulse 2.4s ease-out infinite;
            }
            .icon-badge svg {
                width: 38px;
                height: 38px;
                fill: none;
                stroke: white;
                stroke-width: 1.8;
                stroke-linecap: round;
                stroke-linejoin: round;
            }

            /* Individual icon animations */
            .spark { transform-box: fill-box; transform-origin: center; fill: white; stroke: none; animation: twinkle 2.4s ease-in-out infinite; }
            .spark.small { animation-delay: 0.8s; }
            .bar { transform-box: fill-box; transform-origin: bottom; fill: white; stroke: none; animation: barGrow 2s ease-in-out infinite; }
            .bar.b2 { animation-delay: 0.3s; }
            .bar.b3 { animation-delay: 0.6s; }
            .node { transform-box: fill-box; transform-origin: center; fill: white; stroke: none; animation: nodePulse 2s ease-in-out infinite; }
            .node.n2 { animation-delay: 0.4s; }
            .node.n3 { animation-delay: 0.8s; }
            .node.n4 { animation-delay: 1.2s; }
            .flow { stroke-dasharray: 3 3; animation: dashMove 1.2s linear infinite; }
            .play { transform-box: fill-box; transform-origin: center; fill: white; stroke: none; animation: playPulse 1.8s ease-in-out infinite; }

            /* ---------- Keyframes ---------- */
            @keyframes spin { to { transform: rotate(360deg); } }
            @keyframes heroGlow {
                0%, 100% { filter: drop-shadow(0 0 8px rgba(109, 213, 237, 0.45)); }
                50% { filter: drop-shadow(0 0 22px rgba(109, 213, 237, 0.95)); }
            }
            @keyframes floatY {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-7px); }
            }
            @keyframes ringPulse {
                0% { transform: scale(1); opacity: 0.8; }
                100% { transform: scale(1.5); opacity: 0; }
            }
            @keyframes twinkle {
                0%, 100% { transform: scale(1) rotate(0deg); opacity: 1; }
                50% { transform: scale(0.75) rotate(45deg); opacity: 0.7; }
            }
            @keyframes barGrow {
                0%, 100% { transform: scaleY(0.45); }
                50% { transform: scaleY(1); }
            }
            @keyframes nodePulse {
                0%, 100% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.5); opacity: 0.6; }
            }
            @keyframes dashMove { to { stroke-dashoffset: -12; } }
            @keyframes playPulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.3); }
            }

            /* Respect users who prefer reduced motion */
            @media (prefers-reduced-motion: reduce) {
                .hero-icon svg, .hero-icon .reel, .icon-badge, .icon-badge::after,
                .spark, .bar, .node, .flow, .play { animation: none !important; }
            }
            </style>
        """, unsafe_allow_html=True)

        # Hero section: rotating film reel, title, tagline and key facts
        st.markdown("""
            <div class="hero">
                <div class="hero-icon">
                    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="reelGrad" x1="0" y1="0" x2="1" y2="1">
                                <stop offset="0%" stop-color="#1E3A8A"/>
                                <stop offset="100%" stop-color="#6dd5ed"/>
                            </linearGradient>
                        </defs>
                        <g class="reel">
                            <circle cx="50" cy="50" r="44" fill="rgba(30,58,138,0.25)" stroke="url(#reelGrad)" stroke-width="4"/>
                            <circle cx="50" cy="50" r="9" fill="url(#reelGrad)"/>
                            <circle cx="50" cy="24" r="8" fill="url(#reelGrad)"/>
                            <circle cx="74.7" cy="42" r="8" fill="url(#reelGrad)"/>
                            <circle cx="65.3" cy="71" r="8" fill="url(#reelGrad)"/>
                            <circle cx="34.7" cy="71" r="8" fill="url(#reelGrad)"/>
                            <circle cx="25.3" cy="42" r="8" fill="url(#reelGrad)"/>
                        </g>
                    </svg>
                </div>
                <div>
                    <h1 class="main-title">Movie Magic AI</h1>
                    <p class="hero-sub">An intelligent movie discovery platform that combines machine learning, semantic text embeddings and live metadata to deliver relevant, explainable recommendations.</p>
                    <span class="chip">KMeans · 13 Clusters</span>
                    <span class="chip">BGE Embeddings · 384 Dimensions</span>
                    <span class="chip">Live TMDB Data</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Feature cards, each with its own animated icon
        st.markdown("""
            <div class="feature-grid">
                <div class="movie-card">
                    <div class="icon-badge">
                        <svg viewBox="0 0 24 24">
                            <path class="spark" d="M12 2 L14.2 9.8 L22 12 L14.2 14.2 L12 22 L9.8 14.2 L2 12 L9.8 9.8 Z"/>
                            <path class="spark small" d="M19 2 L19.8 4.2 L22 5 L19.8 5.8 L19 8 L18.2 5.8 L16 5 L18.2 4.2 Z"/>
                        </svg>
                    </div>
                    <h2>Intelligent Movie Discovery</h2>
                    <p>Select a title and receive tailored recommendations derived from genre, director, cast, release year, runtime, rating and audience votes, complemented by semantic analysis of plot descriptions.</p>
                </div>
                <div class="movie-card">
                    <div class="icon-badge">
                        <svg viewBox="0 0 24 24">
                            <rect class="bar" x="3" y="11" width="5" height="10" rx="1"/>
                            <rect class="bar b2" x="9.5" y="5" width="5" height="16" rx="1"/>
                            <rect class="bar b3" x="16" y="8" width="5" height="13" rx="1"/>
                        </svg>
                    </div>
                    <h2>Exploratory Data Analysis</h2>
                    <p>Examine the dataset with confidence. Compare any film against its genre benchmark, review rating distributions and descriptive statistics, and validate data quality through null-value and duplicate-row checks.</p>
                </div>
                <div class="movie-card">
                    <div class="icon-badge">
                        <svg viewBox="0 0 24 24">
                            <line x1="12" y1="6" x2="5.5" y2="17" />
                            <line x1="12" y1="6" x2="18.5" y2="17" />
                            <line x1="5.5" y1="17" x2="18.5" y2="17" />
                            <circle class="node" cx="12" cy="6" r="2.6"/>
                            <circle class="node n2" cx="5.5" cy="17" r="2.6"/>
                            <circle class="node n3" cx="18.5" cy="17" r="2.6"/>
                        </svg>
                    </div>
                    <h2>Normal Search · KMeans Clustering</h2>
                    <p>Movie attributes are standardized and partitioned into 13 clusters using KMeans. Recommendations are drawn from the selected film's cluster and ranked by proximity in feature space.</p>
                </div>
                <div class="movie-card">
                    <div class="icon-badge">
                        <svg viewBox="0 0 24 24">
                            <line class="flow" x1="4" y1="12" x2="12" y2="5" />
                            <line class="flow" x1="4" y1="12" x2="12" y2="19" />
                            <line class="flow" x1="12" y1="5" x2="20" y2="12" />
                            <line class="flow" x1="12" y1="19" x2="20" y2="12" />
                            <line class="flow" x1="12" y1="5" x2="12" y2="19" />
                            <circle class="node" cx="4" cy="12" r="2.2"/>
                            <circle class="node n2" cx="12" cy="5" r="2.2"/>
                            <circle class="node n3" cx="12" cy="19" r="2.2"/>
                            <circle class="node n4" cx="20" cy="12" r="2.2"/>
                        </svg>
                    </div>
                    <h2>Advanced Search · Semantic Embeddings</h2>
                    <p>Plot descriptions are encoded into dense vectors with the BAAI/bge-small-en-v1.5 model. Cosine similarity identifies films with closely related storylines, surfacing relevant titles beyond conventional genre boundaries.</p>
                </div>
                <div class="movie-card">
                    <div class="icon-badge">
                        <svg viewBox="0 0 24 24">
                            <rect x="2.5" y="4.5" width="19" height="15" rx="3"/>
                            <path class="play" d="M10 9 L16 12 L10 15 Z"/>
                        </svg>
                    </div>
                    <h2>Rich Media Integration</h2>
                    <p>Each recommendation is presented with complete metadata, a poster and an official trailer, retrieved in real time from The Movie Database (TMDB) API.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    def eda(self):
        st.markdown('<p class="main-header">📊 Dynamic Movie Insights</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Select a movie to explore its ecosystem within our dataset.</p>', unsafe_allow_html=True)

        # 1. LOAD DATASET AUTOMATICALLY (Internal)
        if "df" not in st.session_state:
            try:
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
            else:
                dup_count = self.df.duplicated().sum()
                if dup_count == 0:
                    st.success("✅ No duplicate rows found.")
                else:
                    st.warning(f"⚠️ {dup_count} duplicate rows detected in global dataset.")

        # 5. DYNAMIC VISUALIZATION
        st.markdown(f"### 📈 How '{selected_movie}' compares to other {movie_data['Genre']} movies")

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(genre_context['Rating'], kde=True, color="#2193b0", ax=ax)
        ax.axvline(movie_data['Rating'], color='green', linestyle='--',
                   label=f'{selected_movie} ({movie_data["Rating"]})')
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

        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(f"**🎭 Genre:** {movie_data.get('Genre', 'N/A')}")
            st.markdown(f"**🎬 Director:** {movie_data.get('Director', 'N/A')}")
        with d2:
            st.markdown(f"**📅 Year:** {movie_data.get('Year', 'N/A')}")
            st.markdown(f"**⏳ Runtime:** {movie_data.get('Runtime (Minutes)', 'N/A')} min")
        with d3:
            st.markdown(f"**⭐ Rating:** {movie_data.get('Rating', 'N/A')}/10")
            st.markdown(f"**🗳️ Votes:** {movie_data.get('Votes', 'N/A')}")

        st.info(f"**📝 Description:** {movie_data.get('Description', 'No description available.')}")

    def kmeans_recs(self, title, clusters, scaled, n):
        """Normal Search: movies from the same cluster, ranked by closeness in feature space."""
        matches = self.df.index[self.df["Title"] == title]
        if len(matches) == 0:
            return []
        pos = matches[0]

        same_cluster = np.where((clusters == clusters[pos]) & (np.arange(len(clusters)) != pos))[0]
        if len(same_cluster) == 0:
            return []

        distances = np.linalg.norm(scaled[same_cluster] - scaled[pos], axis=1)
        best = same_cluster[np.argsort(distances)]

        titles = self.df.loc[best, "Title"]
        titles = titles[titles != title].drop_duplicates()
        return titles.head(n).tolist()

    def render_recommendation(self, i, rec, meta_df):
        """Render one recommendation as a card (details, trailer, poster)."""
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])

            movie_data = get_movie_data(rec)

            if movie_data:
                poster_path = movie_data.get("poster_path")
                trailer_url = get_trailer(movie_data["id"])

                with c1:
                    st.markdown(f"#### `{i}`. {rec}")
                    st.divider()
                    self.show_movie_details(rec, meta_df)

                    if trailer_url:
                        st.video(trailer_url)
                    else:
                        st.info("No trailer found")

                with c2:
                    if poster_path:
                        st.image(TMDB_IMG_BASE + poster_path, use_container_width=True)
                    else:
                        st.warning("Poster not found on TMDB")
            else:
                with c1:
                    st.markdown(f"#### `{i}`. {rec}")
                    st.divider()
                    self.show_movie_details(rec, meta_df)
                st.warning("Movie not found on TMDB")

    def predict(self):
        if "df" not in st.session_state:
            # Auto-load so this page also works without visiting the EDA page first
            try:
                st.session_state.df = pd.read_csv("data/imdb_movie_dataset.csv")
            except FileNotFoundError:
                st.error("Dataset not found! Please ensure your CSV is in the project folder.")
                return

        self.df = st.session_state.df

        st.markdown('<p class="main-header">🍿 Movie Magic Engine</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-driven recommendation with deep-dive metadata explorers.</p>', unsafe_allow_html=True)
        st.markdown("---")

        # Setup inputs
        col_sel, col_count = st.columns([2, 1])

        with col_sel:
            movie_list = self.df["Title"].tolist()
            selected_movie = st.selectbox("Select a Movie you love 🌐", movie_list)

        with col_count:
            recommed_count = st.slider("Recommendations count", 1, 5, 3)

        # --- SECTION 1: SELECTED MOVIE DETAILS ---
        with st.expander(f"✨ View Details for Selected: {selected_movie}", expanded=True):
            self.show_movie_details(selected_movie, self.df)

        # Keep a copy for metadata retrieval before dropping columns
        meta_df = self.df.copy()

        # --- Feature engineering for the KMeans (Normal Search) engine ---
        df_encoded = pd.get_dummies(self.df, columns=["Genre"], drop_first=True, dtype=int)
        for col in ["Director", "Actors"]:
            if col in df_encoded.columns:
                freq_map = self.df[col].value_counts().to_dict()
                df_encoded[col] = df_encoded[col].map(freq_map)

        df_encoded.drop(["Title", "Description"], axis=1, inplace=True, errors="ignore")
        valid_idx = df_encoded.dropna().index
        df_encoded = df_encoded.loc[valid_idx].reset_index(drop=True)
        self.df = self.df.loc[valid_idx].reset_index(drop=True)

        with st.spinner("🧠 ML Engine calibrating clusters..."):
            scaled, clusters = compute_clusters(df_encoded)

        setting = st.sidebar.radio(
            "Select Search Engine Type",
            ["Normal Search", "Advanced Search"],
            help="Normal Search is based on KMeans clustering and Advanced Search on cosine similarity of description embeddings."
        )
        st.sidebar.info("Use Advanced Search for better recommendations.")
        st.markdown("---")

        # --- SECTION 2: RECOMMENDATIONS ---
        if st.button("Generate Recommendations 🚀", use_container_width=True):

            # The selected movie may have been dropped if its row had missing values
            if selected_movie not in self.df["Title"].values:
                st.error("This movie has missing data and cannot be used for recommendations. Please pick another one.")
                return

            if setting == "Advanced Search":
                with st.spinner("Finding similar movies..."):
                    recs = emb_pipeline(selected_movie, self.df, recommed_count)
            else:
                recs = self.kmeans_recs(selected_movie, clusters, scaled, recommed_count)

            if not recs:
                st.error("No matches found. Try changing the Search Engine Type.")
                return

            for i, rec in enumerate(recs, 1):
                self.render_recommendation(i, rec, meta_df)


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
            "📟 About Page": self.run_Home,
            "📊 Data Exploration & Health": self.run_eda,
            "🎬 Movie Recommender Engine": self.run_prediction
        }

        key_select = st.sidebar.selectbox("Go to page", list(options.keys()))
        value_select = options[key_select]

        # Execute page function
        value_select()


# Execution
if __name__ == "__main__":
    app_runner = stream()
    app_runner.app()