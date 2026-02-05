import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process, fuzz
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "Alok8732/Movie_Recommnedation_Artifacts"
TMDB_API_KEY = "9a7f2c468ef72e78bb6f619bea50488b"

@st.cache_resource
def load_model_artifacts():
    """Load the movie dataframe, TF-IDF matrix, and vectorizer from Hugging Face"""
    print("Loading artifacts from Hugging Face...")
    
    # Load Dataframe
    parquet_path = hf_hub_download(repo_id=REPO_ID, filename="movies_fully_cleaned.parquet", repo_type="dataset")
    df = pd.read_parquet(parquet_path)
    df = df.reset_index(drop=True)
    
    # Load TF-IDF Matrix
    matrix_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_matrix.pkl", repo_type="dataset")
    tfidf_matrix = joblib.load(matrix_path)
    
    # Load Vectorizer
    vectorizer_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_vectorizer.pkl", repo_type="dataset")
    tfidf_vectorizer = joblib.load(vectorizer_path)
    
    # Create movie index map
    movie_index_map = pd.Series(df.index, index=df["title"]).drop_duplicates()
    
    print(f"SUCCESS: Loaded {len(df)} movies and a {tfidf_matrix.shape} similarity matrix.")
    return df, tfidf_matrix, tfidf_vectorizer, movie_index_map

def get_poster_url(movie_title):
    """Fetch movie poster from TMDB API"""
    base_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    try:
        response = requests.get(base_url, timeout=5).json()
        if response.get('results'):
            path = response['results'][0]['poster_path']
            if path:
                return f"https://image.tmdb.org/t/p/w500/{path}"
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

def get_recommendations(title, df, tfidf_matrix, movie_index_map, n=12, alpha=0.85, beta=0.15):
    """Generate movie recommendations based on content similarity and popularity"""
    
    # Fuzzy matching to find closest movie title
    all_titles = df['title'].unique()
    best_match, score = process.extractOne(title, all_titles, scorer=fuzz.WRatio)
    
    if score < 60 or len(title.strip()) < 3:
        return None, f"Search term '{title}' is too vague or no close match found."
    
    # Get matrix index
    idx_entry = movie_index_map[best_match]
    idx = idx_entry.iloc[0] if isinstance(idx_entry, pd.Series) else idx_entry
    
    # Compute hybrid score (similarity + popularity)
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    log_pop = np.log1p(df['vote_count'])
    pop_norm = (log_pop - log_pop.min()) / (log_pop.max() - log_pop.min())
    
    hybrid_scores = (alpha * sim_scores) + (beta * pop_norm.values)
    
    # Select top candidates
    k_search = n * 2 
    partition_idx = np.argpartition(hybrid_scores, -k_search)[-k_search:]
    
    # Format results
    res = df.iloc[partition_idx].copy()
    res['similarity'] = sim_scores[partition_idx]
    res['hybrid_score'] = hybrid_scores[partition_idx]
    
    # Remove the searched movie itself
    res = res[res.index != idx]
    
    # Sort by hybrid score and vote count
    res = res.sort_values(by=['hybrid_score', 'vote_count'], ascending=False).head(n)
    
    return res[['title', 'genres', 'vote_count', 'similarity', 'hybrid_score']].reset_index(drop=True), best_match

# ============= STREAMLIT UI =============
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation Engine")
st.markdown("*Powered by TF-IDF Content-Based Filtering + Popularity Boosting*")

# Load model artifacts
with st.spinner("Loading recommendation engine..."):
    df, tfidf_matrix, tfidf_vectorizer, movie_index_map = load_model_artifacts()

# Create two columns for input controls
col1, col2 = st.columns([3, 1])
with col1:
    movie_input = st.text_input("🔍 Enter a movie name:", "Avatar", key="movie_search")
with col2:
    num_recommendations = st.slider("📊 Results:", min_value=6, max_value=24, value=12, step=6)

if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
    with st.spinner(f"Finding movies similar to '{movie_input}'..."):
        results, matched_title = get_recommendations(
            movie_input, 
            df, 
            tfidf_matrix, 
            movie_index_map, 
            n=num_recommendations
        )
    
    if results is None:
        st.error(matched_title)
    else:
        st.success(f"✅ Found matches for: **'{matched_title}'**")
        
        st.markdown("---")
        
        # Display results in a grid
        cols = st.columns(4)
        for i, row in results.iterrows():
            with cols[i % 4]:
                poster_url = get_poster_url(row['title'])
                # FIXED: Changed use_column_width to use_container_width
                st.image(poster_url, use_container_width=True)
                st.markdown(f"**{row['title']}**")
                st.caption(f"🎭 {row['genres']}")
                st.caption(f"⭐ Votes: {int(row['vote_count']):,}")
                st.caption(f"🎯 Match: {row['similarity']:.1%}")
                
                # Progress bar for similarity
                st.progress(row['similarity'])

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(f"""
    This recommendation system uses:
    - **{len(df):,}** movies in database
    - **TF-IDF** for content similarity
    - **Fuzzy matching** for flexible search
    - **Hybrid scoring** (85% similarity + 15% popularity)
    
    ### 🎯 How it works:
    1. Enter a movie title (fuzzy matching enabled)
    2. System finds similar movies based on:
       - Plot, genres, keywords, cast, crew
       - Weighted by popularity
    3. Get personalized recommendations!
    """)
    
    st.header("📊 Dataset Info")
    st.metric("Total Movies", f"{len(df):,}")
    st.metric("Matrix Dimensions", f"{tfidf_matrix.shape[0]:,} x {tfidf_matrix.shape[1]:,}")
    st.metric("Features", f"{tfidf_matrix.shape[1]:,}")
    
    st.markdown("---")
    st.markdown("**💡 Tip:** Try misspelled titles - fuzzy matching will find them!")
