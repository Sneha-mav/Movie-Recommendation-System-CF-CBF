import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import os
from dotenv import load_dotenv

load_dotenv()

# Load Models and Data 

movies = pickle.load(open("models/movies.pkl", "rb"))
similarity = pickle.load(open('models/similarity.pkl', 'rb'))
cf = pickle.load(open("models/cfmodel.pkl", "rb"))

cf_knn_model = cf['cf_knn_model']
user_movie_matrix = cf['user_movie_matrix']
movie_id_to_idx = cf['movie_id_to_idx']
movie_ids = cf['movie_ids']

# Poster Fetch

def fetch_poster(movie_id):
    token = os.getenv("TMDB_API_KEY")
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    headers = {
        "accept": "application/json",
        "Authorization":  f"Bearer {token}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return ""
    data = response.json()
    poster_path = data.get("poster_path")
    return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else ""

#  Content-Based Recommendation

def recommend_movie(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_titles = []
    recommended_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_titles.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_titles, recommended_posters

# Collaborative Filtering Recommendation 

def cf_recommend(movie_name, movies, cf_model_data, n_recs=5):
    try:
        user_movie_matrix = cf_model_data['user_movie_matrix']
        movie_ids = cf_model_data['movie_ids']
        movie_id_to_idx = cf_model_data['movie_id_to_idx']

        movie_titles = movies['title'].tolist()
        movie_title_to_idx = {title: idx for idx, title in enumerate(movie_titles)}

        
        movie_movie_matrix = user_movie_matrix.T  # Shape: (movies, users)

        
        model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_recs + 1)
        model.fit(movie_movie_matrix)

       
        match = process.extractOne(movie_name, movie_titles)
        if not match:
            return [], []

        matched_title = match[0]
        movie_idx = movie_title_to_idx[matched_title]

        distances, indices = model.kneighbors(movie_movie_matrix[movie_idx].reshape(1, -1))

        recommendations = []
        posters = []
        for i in range(1, n_recs + 1): 
            rec_idx = indices.flatten()[i]
            rec_title = movie_titles[rec_idx]
            movie_row = movies[movies['title'] == rec_title]
            if not movie_row.empty:
                tmdb_id = movie_row.iloc[0]['movie_id']
                recommendations.append(rec_title)
                posters.append(fetch_poster(tmdb_id))

        return recommendations, posters

    except Exception as e:
        st.error(f"Collaborative Filtering Error: {str(e)}")
        return [], []

# Streamlit 

st.title("🎬 Movie Recommendation System")

tab1, tab2 = st.tabs(["🔍 Content-Based", "🤝 Collaborative Filtering"])

# Tab1
with tab1:
    st.subheader("Recommend Similar Movies")
    selected_movie_cb = st.selectbox("Select a movie:", movies["title"].values, key="cb_select")

    if st.button("Recommend", key="cb_btn"):
        titles, posters = recommend_movie(selected_movie_cb)
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.text(titles[i] if i < len(titles) else "N/A")
                st.image(posters[i] if i < len(posters) else None)

# Tab2
with tab2:
    st.subheader("Collaborative Filtering Based Recommendation")

    selected_movie_cf = st.selectbox("Select a movie:", movies["title"].values, key="cf_select")

    if st.button("Get CF Recommendations", key="cf_btn"):
        titles_cf, posters_cf = cf_recommend(selected_movie_cf, movies, cf, n_recs=5)
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.text(titles_cf[i] if i < len(titles_cf) else "No recommendation")
                st.image(posters_cf[i] if i < len(posters_cf) else None)

