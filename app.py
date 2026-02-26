import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MovAi - Film Rehberi", layout="wide")

# --- TASARIM GÜNCELLEMESİ ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%); color: white; }
    h1 { color: #00FFFF !important; text-align: center; font-family: 'Arial Black'; font-size: 3rem !important; margin-bottom: 30px; }
    
    .film-baslik { 
        color: #FFFFFF !important; 
        font-size: 16px !important; 
        font-weight: bold; 
        text-align: center; 
        padding: 5px;
        min-height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2px;
    }

    /* IMDb Puanı Stili */
    .imdb-puan {
        color: #FFD700 !important; /* Altın Sarısı */
        font-weight: bold;
        text-align: center;
        font-size: 14px;
        margin-bottom: 10px;
    }
    
    .stImage { margin-bottom: 30px; border-radius: 10px; }

    div.stButton > button { 
        background-color: #008B8B; 
        color: white; 
        border: 2px solid #00FFFF; 
        border-radius: 10px; 
        width: 100%; 
        height: 3.5rem; 
        font-size: 1.2rem;
    }
    div.stButton > button:hover { background-color: #00FFFF; color: black; box-shadow: 0 0 15px #00FFFF; }
    </style>
    """, unsafe_allow_html=True)

# 1. Verileri yükle
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# 2. Benzerlik matrisini hesapla
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# 3. Poster ve Puan Çekme Fonksiyonu
def fetch_info(movie_id):
    api_key = "71688304490ebfafbeb6e454a722ebc4" 
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=tr-TR"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        vote_average = data.get('vote_average', 0)
        
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else "https://via.placeholder.com/500x750/1B2735/00FFFF/?text=Resim+Yok"
        return full_path, round(vote_average, 1)
    except:
        return "https://via.placeholder.com/500x750/1B2735/00FFFF/?text=Hata", 0

# --- ARAYÜZ ---
st.title('🚀 MovAi: Sonsuz Film Keşfi')

selected_movie = st.selectbox('Benzerlerini bulmak istediğin filmi seç:', movies['title'].values)

if st.button('Tüm Benzer Filmleri Getir'):
    movie_index = movies[movies['title'].str.lower() == selected_movie.lower()].index[0]
    distances = similarity[movie_index]
    
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:21]
    
    st.write(f"### 🌌 '{selected_movie}' İçin Önerilenler:")
    st.write("---")
    
    for i in range(0, len(movies_list), 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(movies_list):
                idx = movies_list[i + j][0]
                with cols[j]:
                    m_id = movies.iloc[idx].movie_id
                    poster, rating = fetch_info(m_id)
                    
                    # Başlık ve Puan
                    st.markdown(f"<div class='film-baslik'>{movies.iloc[idx].title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='imdb-puan'>⭐ {rating} / 10</div>", unsafe_allow_html=True)
                    st.image(poster, use_container_width=True)

st.write("---")
st.markdown("<p style='text-align: center; color: #888;'>MovAi - IMDb Puan Destekli Öneri Sistemi</p>", unsafe_allow_html=True)
