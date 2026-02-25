import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Küçük boyutlu veriyi yükle (25MB sınırına takılmayan dosya)
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# 2. Benzerlik matrisini site açılırken anında hesapla (Dev dosyadan kurtulduk)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# 3. Öneri Fonksiyonu
def recommend(movie):
    try:
        movie_index = movies[movies['title'].str.lower() == movie.lower()].index[0]
    except IndexError:
        return ["Film bulunamadı."]
        
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# 4. Web Arayüzü Tasarımı
st.title('🎬 Film Öneri Sistemi')

selected_movie_name = st.selectbox(
    'Hangi filme benzer bir şeyler izlemek istersin?',
    movies['title'].values
)

if st.button('Önerileri Göster'):
    recommendations = recommend(selected_movie_name)
    
    st.write("### İşte senin için seçtiklerimiz:")
    for i in recommendations:
        st.write("- " + i)