import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

god= pd.read_csv("movies.csv")
x = CountVectorizer()
y = x.fit_transform(god['genres'])
how_we_know = cosine_similarity(y)

def recommend(movie):
    if movie not in god['title'].values:
        print("\n Movie not found! Please try another movie.\n")
        return
        
    index = god[god['title'] == movie].index[0]
    distances = list(enumerate(how_we_know[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    print(f"\n🎬 Recommended movies for '{movie}':\n")
    for i in movies_list:
        print(god.iloc[i[0]].title)

movie_name = input("Enter a movie name: ")
recommend(movie_name)