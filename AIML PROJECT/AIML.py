import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# the code and movies.csv should be in the same folder for the code to work
god= pd.read_csv("movies.csv")
#count vectorizer is used to convert the text data into a matrix of token counts
#cosine similarity is used to calculate the similarity between the movies based on the genres of the movies. The higher the cosine similarity, the more similar the movies are.
#the code will recommend movies based on the genres of the movies you have searched for before
x = CountVectorizer()
#fit_transform is used to fit the model and transform the data into a matrix of token counts

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

movie_that_you_like= input("Enter a movie name: ")
recommend(movie_that_you_like)