import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MovAi - Film Rehberi", layout="wide")

# --- TASARIM (CSS) - Neon Etkisi Kaldırıldı ve Yazılar Büyütüldü ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
        color: white;
    }
    
    /* Başlık: Neon kaldırıldı, daha net ve büyük */
    h1 {
        color: #00FFFF !important;
        text-align: center;
        font-family: 'Arial Black', Gadget, sans-serif;
        font-size: 3rem !important;
        padding-bottom: 20px;
    }
    
    /* Seçim kutusu etiketleri */
    .stSelectbox label {
        color: #00FFFF !important;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    /* Film İsimleri: Afişlerin üzerindeki metni büyütüp kalınlaştırdık */
    .film-baslik {
        color: #FFFFFF !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        text-align: center;
        min-height: 55px;
        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1.2;
        margin-bottom: 5px;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Buton Tasarımı */
    div.stButton > button:first-child {
        background-color: #008B8B;
        color: white;
        border: 2px solid #00FFFF;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
        height: 3.5rem;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #00FFFF;
        color: black;
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

# 3. Poster Çekme Fonksiyonu
def fetch_poster(movie_id):
    api_key = "71688304490ebfafbeb6e454a722ebc4" # Kendi anahtarını buraya koymayı unutma
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=tr-TR"
        data = requests.get(url).json()
        poster_path = data['poster_path']
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return "https://via.placeholder.com/500x750/1B2735/00FFFF/?text=Resim+Yok"

# 4. Kategori Sözlüğü
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
        
        rec_names = []
        rec_posters = []
        eng_genre = kategori_sozlugu[tr_genre]
        
        for i in movies_list[1:]:
            current_tags = movies.iloc[i[0]].tags.lower()
            if eng_genre == "Tümü" or eng_genre in current_tags:
                rec_names.append(movies.iloc[i[0]].title)
                rec_posters.append(fetch_poster(movies.iloc[i[0]].movie_id))
            if len(rec_names) == 5: break
        return rec_names, rec_posters
    except:
        return [], []

# --- ARAYÜZ ---
st.title('🚀 MovAi: Akıllı Film Rehberi')

c1, c2 = st.columns(2)
with c1:
    selected_movie = st.selectbox('Benzerini bulmak istediğin film:', movies['title'].values)
with c2:
    selected_genre = st.selectbox('Kategori seç (Opsiyonel):', list(kategori_sozlugu.keys()))

if st.button('Önerileri Sırala'):
    names, posters = recommend(selected_movie, selected_genre)
    
    if names:
        st.write("---")
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                # Film ismini büyük ve okunaklı hale getiren özel sınıf
                st.markdown(f"<div class='film-baslik'>{names[idx]}</div>", unsafe_allow_html=True)
                st.image(posters[idx], use_container_width=True)
    else:
        st.warning("Eşleşen sonuç bulunamadı.")

