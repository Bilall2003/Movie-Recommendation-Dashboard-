import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

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
                <h2>🎬 Cinematic Intelligence</h2>
                <p>Experience the next generation of movie discovery. Our AI analyzes thousands of data points including genres, directors, and cast chemistry to find your next favorite film.</p>
            </div>
            
            <div class="movie-card">
                <img class="card-icon" src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png">
                <h2>📊 Data-Driven Discovery</h2>
                <p>Upload your own dataset and watch as the engine automatically cleans and transform dataset into logical groupings.</p>
            </div>

            <div class="movie-card">
                <img class="card-icon" src="https://cdn-icons-png.flaticon.com/512/1491/1491468.png">
                <h2>🧠 Hybrid Recommendation Engine</h2>
                <p>Switch between standard Genre-matching or our proprietary Hybrid Engine that uses ML model to find "hidden gem" matches outside of standard categories.</p>
            </div>
        """, unsafe_allow_html=True)

        
    def eda(self):
        st.markdown('<p class="main-header">📊 Data Exploratory Hub</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Upload, clean, and understand your dataset in seconds.</p>', unsafe_allow_html=True)
        
        # Top Upload Card
        with st.container():
            self.uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type="csv")

        if self.uploaded_file is not None:  
            st.session_state.df = pd.read_csv(self.uploaded_file)
            self.df = st.session_state.df

            place_holder = st.empty()
            place_holder.success("🎉 File Uploaded Successfully!")
            time.sleep(1)
            place_holder.empty()
            
        elif "df" in st.session_state:
            self.df = st.session_state.df
            st.toast("Using previously cached dataset ✅")

        # Display Section if DataFrame is present
        if "df" in st.session_state:
            st.markdown("---")
            
            # Preview and Data Types in columns
            col_preview, col_types = st.columns([2, 1])
            
            with col_preview:
                st.subheader(" Dataset Quick Preview")
                st.dataframe(self.df.head(6), use_container_width=True)
                
            with col_types:
                st.subheader("🏷️ Columns & Data Types")
                # Wrap dtypes in a clean expander to save space
                with st.expander("Show Data Types", expanded=True):
                    dtypes_df = pd.DataFrame(self.df.dtypes.astype(str), columns=["Data Type"])
                    st.dataframe(dtypes_df, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🛠️ Data Health Dashboard")
            
            # Action selection
            data_options = ["Select an Analysis", "🔍 Analyze Null Values", "🎯 Analyze Duplicate Rows"]
            user_choice = st.selectbox("Deep dive into data health:", data_options)
            
            if user_choice == data_options[1]:
                st.markdown("#### 🔍 Missing Values Analysis & Resolution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("📌 **Before Cleaning**")
                    percent_nan = self.df.isnull().sum() / len(self.df) * 100
                    percent_nan = percent_nan[percent_nan > 0]
                    
                    if not percent_nan.empty:
                        fig1, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(x=percent_nan.index, y=percent_nan.values, palette="flare", ax=ax)
                        plt.xticks(rotation=45)
                        plt.ylabel("% Missing")
                        st.pyplot(fig1)
                    else:
                        st.success("Clean Dataset! 0% missing values detected.")
                        
                with col2:
                    st.info("🧹 **After Automatic Cleaning**")
                    # Clean the data in session state directly
                    st.session_state.df = self.df.dropna()
                    self.df = st.session_state.df
                    
                    percent_nan_after = self.df.isnull().sum() / len(self.df) * 100
                    percent_nan_after = percent_nan_after[percent_nan_after > 0]
                    
                    fig2, ax = plt.subplots(figsize=(8, 4))
                    if not percent_nan_after.empty:
                        sns.barplot(x=percent_nan_after.index, y=percent_nan_after.values, palette="viridis", ax=ax)
                    else:
                        # Draw empty or clean chart
                        ax.text(0.5, 0.5, "0% Missing Values Left", horizontalalignment='center', verticalalignment='center', fontsize=12, color='green')
                        ax.set_axis_off()
                    st.pyplot(fig2)
                    
            elif user_choice == data_options[2]:
                st.markdown("#### 🎯 Duplicate Rows Analysis")
                
                percent_dup = self.df.duplicated().sum() / len(self.df) * 100
                percent_unique = 100 - percent_dup

                # Metric highlight
                met1, met2 = st.columns(2)
                met1.metric(label="Duplicate Percentage", value=f"{percent_dup:.2f}%")
                met2.metric(label="Unique Rows Percentage", value=f"{percent_unique:.2f}%")

                labels = ["Duplicate Rows", "Unique Rows"]
                sizes = [percent_dup, percent_unique]
                colors = ["#F43F5E", "#10B981"]

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(sizes, labels=labels, autopct="%.1f%%", startangle=90, colors=colors, explode=(0.05, 0))
                ax.axis("equal") 
                st.pyplot(fig)
                
            st.markdown("---")
            st.markdown("### 📈 Statistical Summary")
            with st.expander("View Descriptive Analysis Table", expanded=True):
                st.dataframe(
                    self.df.describe().style.background_gradient(cmap="Blues"), 
                    use_container_width=True
                )
                
        else:
            st.info("👆 Please upload a CSV file to get started.")


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
        if "df" in st.session_state:
            self.df = st.session_state.df
            
            st.markdown('<p class="main-header">🍿 Movie Magic Engine</p>', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">AI-driven clustering with deep-dive metadata explorers.</p>', unsafe_allow_html=True)
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

            # Data Processing (ML pipeline)
            # We keep a copy for metadata retrieval before dropping columns
            meta_df = self.df.copy() 
            
            df_encoded = pd.get_dummies(self.df, columns=["Genre"], drop_first=True, dtype=int)
            col2 = ["Director", "Actors"]
            for col in col2:
                if col in df_encoded.columns:
                    freq_map = self.df[col].value_counts().to_dict()
                    df_encoded[col] = df_encoded[col].map(freq_map)
            
            df_encoded.drop(["Title", "Description"], axis=1, inplace=True, errors="ignore")
            valid_idx = df_encoded.dropna().index
            df_encoded = df_encoded.loc[valid_idx].reset_index(drop=True)
            self.df = self.df.loc[valid_idx].reset_index(drop=True)
            
            with st.spinner("🧠 ML Engine calibrating clusters..."):
                operation = make_pipeline(StandardScaler(), KMeans(random_state=101, n_init=10, n_clusters=13))
                operation.fit(df_encoded)
                df_encoded["clusters"] = operation.predict(df_encoded)

            setting = st.sidebar.radio("Select Search Engine Type", ["Normal Search", "Hybrid Intelligence Engine"])
            st.sidebar.info("Use HIE for better recommendations....")
            st.markdown("---")

            # --- SECTION 2: RECOMMENDATIONS ---
            if st.button("Generate Recommendations 🚀", use_container_width=True):
                # Filter Logic
                if setting == "Hybrid Intelligence Engine":
                    selected_cluster = df_encoded.loc[self.df["Title"] == selected_movie, "clusters"].values[0]
                    selected_genre = self.df.loc[self.df["Title"] == selected_movie, "Genre"].values[0]
                    mask = (df_encoded["clusters"] == selected_cluster) & (self.df["Genre"] == selected_genre)
                else:
                    selected_genre = self.df.loc[self.df["Title"] == selected_movie, "Genre"].values[0]
                    mask = (self.df["Genre"] == selected_genre)

                similar_movies = self.df.loc[mask, "Title"]
                similar_movies = similar_movies[similar_movies != selected_movie]

                if not similar_movies.empty:
                    st.balloons()
                    st.success(f"Top {min(len(similar_movies), recommed_count)} Recommendations:")
                    
                    recs = similar_movies.sample(min(len(similar_movies), recommed_count)).tolist()
                    
                    for i, rec in enumerate(recs, 1):
                        # Create a card-like container for each recommendation
                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"#### `{i}`. {rec}")
                                st.divider()
                                self.show_movie_details(rec, meta_df)

                else:
                    st.error("No matches found. Try changing the Search Engine Type.")
        else:
            st.warning("⚠️ Please upload a dataset on the 'Data Exploration' page first.")

class stream(predicter):
    
    def run_Home(self):
        self.home()
    
    def run_eda(self):
        self.eda()
        
    def run_prediction(self):
        self.predict()
        
    def app(self):
        st.sidebar.markdown("### Menu & Controls")
        
        # Shiny connected badge
        st.sidebar.markdown(
            """
            <div style='background-color:green; color:white; padding:8px 12px; border-radius:15px; font-weight:600; font-size:13px; text-align:center; border: 1px solid white; margin-bottom:15px;'>
               💚 Connected to Weaviate
            </div>
            """, 
            unsafe_allow_html=True
        )

        options = {
            "📟 About Page":self.run_Home,
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