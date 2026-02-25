import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MovAi - Film Rehberi", layout="wide")

# --- TASARIM GÜNCELLEMESİ (Hizalama ve Boşluklar) ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%); color: white; }
    h1 { color: #00FFFF !important; text-align: center; font-family: 'Arial Black'; font-size: 3rem !important; margin-bottom: 30px; }
    
    /* Film Başlığı: Afişin tam üstünde, büyük ve boşluklu */
    .film-baslik { 
        color: #00FFFF !important; 
        font-size: 16px !important; 
        font-weight: bold; 
        text-align: center; 
        padding: 10px 5px; /* Yazının etrafına iç boşluk */
        min-height: 65px; /* Uzun isimler için sabit yükseklik */
        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1.3;
        background: rgba(0, 255, 255, 0.05); /* Çok hafif bir arka plan ki isimler belirginleşsin */
        border-radius: 10px 10px 0 0; /* Üst köşeleri yuvarla */
        margin-bottom: 5px; /* Afişle arasına boşluk */
    }
    
    /* Afişlerin etrafına biraz nefes aldıracak boşluk */
    .stImage {
        margin-bottom: 30px; /* Bir sonraki satırdaki isimle arasına boşluk */
        border-radius: 0 0 10px 10px;
    }

    div.stButton > button { 
        background-color: #008B8B; 
        color: white; 
        border: 2px solid #00FFFF; 
        border-radius: 10px; 
        width: 100%; 
        height: 3.5rem; 
        font-size: 1.2rem;
        margin-top: 20px;
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

# 3. Poster Çekme Fonksiyonu
def fetch_poster(movie_id):
    api_key = "71688304490ebfafbeb6e454a722ebc4" 
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=tr-TR"
        data = requests.get(url).json()
        poster_path = data['poster_path']
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return "https://via.placeholder.com/500x750/1B2735/00FFFF/?text=Resim+Yok"

# --- ARAYÜZ ---
st.title('🚀 MovAi: Sonsuz Film Keşfi')

selected_movie = st.selectbox('Benzerlerini bulmak istediğin filmi seç:', movies['title'].values)

if st.button('Tüm Benzer Filmleri Getir'):
    movie_index = movies[movies['title'].str.lower() == selected_movie.lower()].index[0]
    distances = similarity[movie_index]
    
    # Sayfa hızı için en benzer 20 filmi alıyoruz
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:21]
    
    st.write(f"### 🌌 '{selected_movie}' İçin Keşfedilen Galaksiler:")
    st.write("---")
    
    # 5'li Izgara Yapısı
    for i in range(0, len(movies_list), 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(movies_list):
                idx = movies_list[i + j][0]
                with cols[j]:
                    # İsim ve Afiş
                    st.markdown(f"<div class='film-baslik'>{movies.iloc[idx].title}</div>", unsafe_allow_html=True)
                    st.image(fetch_poster(movies.iloc[idx].movie_id), use_container_width=True)

st.write("---")
st.markdown("<p style='text-align: center; color: #888;'>MovAi - Yapay Zeka Destekli Film Öneri Sistemi</p>", unsafe_allow_html=True)
