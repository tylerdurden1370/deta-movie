import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Veriyi yükle
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# 2. Benzerlik matrisini anında hesapla
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# YENİ: İnternetten film afişini çeken fonksiyon
def fetch_poster(movie_id):
    # API anahtarını buraya yapıştır!
    api_key = "71688304490ebfafbeb6e454a722ebc4" 
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# 3. Öneri Fonksiyonu (Artık hem isimleri hem afişleri döndürüyor)
def recommend(movie):
    try:
        movie_index = movies[movies['title'].str.lower() == movie.lower()].index[0]
    except IndexError:
        return ["Film bulunamadı."], []
        
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_movie_posters = []
    
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        # Filmin ID'sini kullanarak afişini çekiyoruz
        recommended_movie_posters.append(fetch_poster(movie_id))
        
    return recommended_movies, recommended_movie_posters

# --- YENİ WEB ARAYÜZÜ TASARIMI ---
st.title('🎬 Film Öneri Sistemi')

selected_movie_name = st.selectbox(
    'Hangi filme benzer bir şeyler izlemek istersin?',
    movies['title'].values
)

if st.button('Önerileri Göster'):
    names, posters = recommend(selected_movie_name)
    
    st.write("### İşte senin için seçtiklerimiz:")
    
    # Ekranı 5 eşit sütuna bölüyoruz
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
