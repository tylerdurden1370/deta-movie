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
        color: #00FFFF !important; 
        font-size: 14px !important; 
        font-weight: bold; 
        text-align: center; 
        padding: 8px 5px;
        min-height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1.2;
        background: rgba(0, 255, 255, 0.08);
        border-radius: 10px 10px 0 0;
    }

    .film-puan {
        background: #00FFFF;
        color: #000;
        font-size: 12px;
        font-weight: bold;
        text-align: center;
        border-radius: 0 0 0 0;
        padding: 2px;
    }
    
    .stImage {
        margin-bottom: 25px;
        border-radius: 0 0 10px 10px;
        transition: transform .2s;
    }
    .stImage:hover { transform: scale(1.05); }

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

# 1. Verileri yükle (Burada 'score' ve 'vote_average' sütunlarının olduğunu varsayıyoruz)
@st.cache_resource # Sayfa her yenilendiğinde tekrar yükleyip yavaşlamasın diye önbelleğe alıyoruz
def load_data():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies_df = pd.DataFrame(movies_dict)
    
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies_df['tags']).toarray()
    sim = cosine_similarity(vectors)
    return movies_df, sim

movies, similarity = load_data()

# 2. Poster Çekme Fonksiyonu
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
    movie_index = movies[movies['title'] == selected_movie].index[0]
    distances = similarity[movie_index]
    
    # Adım A: Önce en benzer 50 filmi "aday" olarak alıyoruz
    similar_candidates = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:51]
    candidate_indices = [i[0] for i in similar_candidates]
    
    # Adım B: Bu 50 filmi 'score' (Ağırlıklı IMDB puanı) değerine göre sıralayıp ilk 20'yi alıyoruz
    recommendations_df = movies.iloc[candidate_indices].sort_values('score', ascending=False).head(20)
    
    st.write(f"### 🌌 '{selected_movie}' Sevenler İçin Kalite Odaklı Öneriler:")
    st.write("---")
    
    # 5'li Izgara Yapısı
    reco_list = recommendations_df.to_dict('records') # İşlem kolaylığı için listeye çevirdik
    
    for i in range(0, len(reco_list), 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(reco_list):
                item = reco_list[i + j]
                with cols[j]:
                    # İsim, Puan ve Afiş
                    st.markdown(f"<div class='film-baslik'>{item['title']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='film-puan'>⭐ IMDB: {item['vote_average']}</div>", unsafe_allow_html=True)
                    st.image(fetch_poster(item['movie_id']), use_container_width=True)

st.write("---")
st.markdown("<p style='text-align: center; color: #888;'>MovAi - Yapay Zeka Destekli Film Öneri Sistemi</p>", unsafe_allow_html=True)

st.write("---")
st.markdown("<p style='text-align: center; color: #888;'>MovAi - Yapay Zeka Destekli Film Öneri Sistemi</p>", unsafe_allow_html=True)


