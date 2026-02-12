import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.sparse import load_npz
from difflib import get_close_matches

# Custom CSS with animations
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .header {
            background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            animation: fadeIn 1s;
        }
        
        .footer {
            background: linear-gradient(135deg, #4776E6 0%, #8E54E9 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 2rem;
            text-align: center;
            animation: slideUp 1s;
        }
        
        .recommendation-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        
        .recommendation-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.2);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideUp {
            from { transform: translateY(50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 50px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(110, 72, 170, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return {
        'movies': joblib.load('models/movies.joblib'),
        'knn': joblib.load('models/knn_model.joblib'),
        'tfidf_matrix': load_npz('models/tfidf_matrix.npz')
    }

def get_recommendations(movie_title, models, n=5):
    movies = models['movies']
    knn = models['knn']
    tfidf_matrix = models['tfidf_matrix']
    
    matches = get_close_matches(movie_title.lower(), 
                              movies['title'].str.lower(), 
                              n=1, cutoff=0.3)
    if not matches:
        return None, None
    
    movie_idx = movies[movies['title'].str.lower() == matches[0]].index[0]
    distances, indices = knn.kneighbors(tfidf_matrix[movie_idx])
    
    return movies.iloc[indices[0][1:n+1]], movies.iloc[movie_idx]

def show_genre_wordcloud(recommendations):
    all_genres = [genre for sublist in recommendations['genres'] for genre in sublist]
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='viridis').generate(' '.join(all_genres))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Genre Word Cloud', pad=20, fontsize=16)
    st.pyplot(fig)

def show_genre_distribution(recommendations):
    all_genres = [genre for sublist in recommendations['genres'] for genre in sublist]
    genre_counts = pd.Series(all_genres).value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='magma')
    ax.set_title('Genre Distribution', pad=20, fontsize=16)
    ax.set_xlabel('Count')
    ax.set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig)

def show_year_distribution(recommendations):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=recommendations, x='year', bins=15, 
                 kde=True, color='purple')
    ax.set_title('Release Year Distribution', pad=20, fontsize=16)
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def main():
    inject_custom_css()
    
    # Animated Header
    st.markdown("""
    <div class="header">
        <h1 style="margin:0; font-size:2.5rem;">🎬 MovieMagic AI</h1>
        <p style="margin:0; font-size:1.1rem;">Discover your perfect movie match</p>
    </div>
    """, unsafe_allow_html=True)
    
    models = load_models()
    movie_title = st.text_input("🔍 Enter a movie you enjoy:", placeholder="The Dark Knight, Inception...")
    
    if st.button("Get Recommendations"):
        if movie_title:
            with st.spinner('✨ Analyzing your taste...'):
                recommendations, original = get_recommendations(movie_title, models, 5)
                
                if recommendations is not None:
                    st.success(f"### Because you liked **{original['title']}** ({original['year']})")
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("#### Recommended Movies")
                        for _, row in recommendations.iterrows():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>{row['title']} ({row['year']})</h4>
                                <p>📌 {', '.join(row['genres'])}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        st.markdown("#### Visual Insights")
                        show_genre_wordcloud(recommendations)
                        show_genre_distribution(recommendations)
                        show_year_distribution(recommendations)
                else:
                    st.error("Movie not found. Try being more specific or check spelling.")
        else:
            st.warning("Please enter a movie title to get recommendations")
    
    # Animated Footer
    st.markdown("""
    <div class="footer">
        <h3 style="margin:0">Developed with ❤️ by AhmedAI</h3>
        <p style="margin:0.5rem 0 0;">Movie Recommendation Engine © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()