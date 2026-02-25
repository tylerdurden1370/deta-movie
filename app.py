import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MovAi - Film Rehberi", layout="wide")

# --- TASARIM (CSS) ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%); color: white; }
    h1 { color: #00FFFF !important; text-align: center; font-family: 'Arial Black'; font-size: 3rem !important; }
    .film-baslik { color: #FFFFFF !important; font-size: 18px !important; font-weight: 800 !important; text-align: center; min-height: 55px; line-height: 1.2; font-family: 'Segoe UI'; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { color: #00FFFF !important; font-size: 20px; font-weight: bold; }
    div.stButton > button { background-color: #008B8B; color: white; border: 2px solid #00FFFF; border-radius: 10px; width: 100%; height: 3rem; }
    div.stButton > button:hover { background-color: #00FFFF; color: black; }
    </style>
    """, unsafe_allow_html=True)

# 1. Verileri yükle
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# 2. Benzerlik matrisini hesapla
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# 3. Poster Çekme
def fetch_poster(movie_id):
    api_key = "71688304490ebfafbeb6e454a722ebc4
" # Burayı doldurmayı unutma
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=tr-TR"
        data = requests.get(url).json()
        poster_path = data['poster_path']
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return "https://via.placeholder.com/500x750/1B2735/00FFFF/?text=Resim+Yok"

# --- ARAYÜZ ---
st.title('🚀 MovAi: Akıllı Film Rehberi')

# SEKME SİSTEMİ: İkisinden biri seçilebilsin diye ayırıyoruz
tab1, tab2 = st.tabs(["🔍 Filme Benzer Öner", "📂 Kategoriye Göre Keşfet"])

with tab1:
    selected_movie = st.selectbox('Öneri almak istediğin ana filmi seç:', movies['title'].values)
    if st.button('Benzerlerini Bul'):
        movie_index = movies[movies['title'].str.lower() == selected_movie.lower()].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                st.markdown(f"<div class='film-baslik'>{movies.iloc[movies_list[idx][0]].title}</div>", unsafe_allow_html=True)
                st.image(fetch_poster(movies.iloc[movies_list[idx][0]].movie_id), use_container_width=True)

with tab2:
    kategoriler = {"Aksiyon": "action", "Macera": "adventure", "Komedi": "comedy", "Korku": "horror", "Bilim Kurgu": "science fiction", "Dram": "drama"}
    selected_genre = st.selectbox('Hangi türde film istersin?', list(kategoriler.keys()))
    
    if st.button('Türün En İyilerini Getir'):
        eng_genre = kategoriler[selected_genre]
        # Kategoriye göre basit bir filtreleme
        genre_movies = movies[movies['tags'].str.contains(eng_genre)].head(5)
        
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                st.markdown(f"<div class='film-baslik'>{genre_movies.iloc[idx].title}</div>", unsafe_allow_html=True)
                st.image(fetch_poster(genre_movies.iloc[idx].movie_id), use_container_width=True)
