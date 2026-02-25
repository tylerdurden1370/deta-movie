import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MovAi - Film Keşfi", layout="wide")

# --- TASARIM (CSS) ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
        color: white;
    }
    h1 {
        color: #00FFFF !important;
        text-shadow: 0 0 15px #00FFFF;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stSelectbox label {
        color: #00FFFF !important;
        font-weight: bold;
    }
    div.stButton > button:first-child {
        background-color: #008B8B;
        color: white;
        border: 2px solid #00FFFF;
        border-radius: 30px;
        font-weight: bold;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #00FFFF;
        color: black;
        box-shadow: 0 0 30px #00FFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# 1. Verileri yükle
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# 2. Benzerlik matrisini hesapla
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# 3. TMDB API Poster Çekme
def fetch_poster(movie_id):
    api_key = "71688304490ebfafbeb6e454a722ebc4" # Buraya kendi anahtarını yapıştır
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=tr-TR"
        data = requests.get(url).json()
        poster_path = data['poster_path']
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return "https://via.placeholder.com/500x750/000000/00FFFF/?text=Afiş+Yok"

# 4. Kategori Sözlüğü (Türkçe -> İngilizce Tag Eşleşmesi)
kategori_sozlugu = {
    "Tümü": "Tümü",
    "Aksiyon": "action",
    "Macera": "adventure",
    "Animasyon": "animation",
    "Komedi": "comedy",
    "Korku": "horror",
    "Bilim Kurgu": "science fiction",
    "Romantik": "romance",
    "Gerilim": "thriller",
    "Dram": "drama"
}

# 5. Öneri Fonksiyonu
def recommend(movie, tr_genre):
    try:
        movie_index = movies[movies['title'].str.lower() == movie.lower()].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
        
        recommended_movies = []
        recommended_movie_posters = []
        
        eng_genre = kategori_sozlugu[tr_genre]
        
        for i in movies_list[1:]:
            current_tags = movies.iloc[i[0]].tags.lower()
            
            if eng_genre == "Tümü" or eng_genre in current_tags:
                recommended_movies.append(movies.iloc[i[0]].title)
                recommended_movie_posters.append(fetch_poster(movies.iloc[i[0]].movie_id))
            
            if len(recommended_movies) == 5:
                break
                
        return recommended_movies, recommended_movie_posters
    except:
        return [], []

# --- WEB ARAYÜZÜ ---
st.title('🚀 MovAi: Akıllı Film Rehberi')

col1, col2 = st.columns(2)

with col1:
    selected_movie_name = st.selectbox(
        'Hangi filme benzer öneriler istersin?',
        movies['title'].values
    )

with col2:
    secilen_kategori = st.selectbox(
        'Bir tür seçmek ister misin?',
        list(kategori_sozlugu.keys())
    )

st.write("---")

if st.button('Önerileri Getir'):
    names, posters = recommend(selected_movie_name, secilen_kategori)
    
    if names:
        st.write(f"### 🌌 {secilen_kategori} Kategorisindeki Öneriler:")
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                st.markdown(f"<p style='color:#00FFFF; font-size:14px; font-weight:bold; min-height:45px;'>{names[idx]}</p>", unsafe_allow_html=True)
                st.image(posters[idx], use_container_width=True)
    else:
        st.warning("Aradığınız kriterlerde uygun film bulunamadı.")
