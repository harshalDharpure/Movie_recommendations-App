import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the dummy dataset
@st.cache_data
def load_data():
    return pd.read_csv('movie_data.csv')

df = load_data()

# Compute cosine similarity matrix for movie recommendations
cosine_sim = cosine_similarity(df[['Rating']], df[['Rating']])

# Function to recommend movies
def recommend_movies(movie_title, num_recommendations=5):
    idx = df[df['Title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return df['Title'].iloc[movie_indices]

# Streamlit UI
st.title('Movie Recommendation App')
st.sidebar.title('Recommendation Settings')

# Sidebar input for movie selection
selected_movie = st.sidebar.selectbox('Select a Movie', df['Title'])

# Number of recommendations to display
num_recommendations = st.sidebar.slider('Number of Recommendations', 1, 10, 5)

if st.button('Get Recommendations'):
    recommended_movies = recommend_movies(selected_movie, num_recommendations)
    st.subheader(f'Recommended Movies for {selected_movie}:')
    for movie in recommended_movies:
        st.write(movie)

# Display the dataset
st.dataframe(df)
