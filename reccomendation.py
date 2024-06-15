import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Loading datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Creating pivot table
final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
final_dataset.fillna(0, inplace=True)

# Filtering movies and users with low interaction
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 50].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Converting to CSR matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Training KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Recommendation function
def get_movie_recommendation(movie_name):
    n_movies_to_recommend = 10
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, na=False)]  # Case-insensitive search
    
    if movie_list.empty:
        return "No movies found. Please check your input"
    
    movie_id = movie_list.iloc[0]['movieId']
    
    if movie_id not in final_dataset['movieId'].values:
        return "Movie not found in final dataset"
    
    movie_idx = final_dataset[final_dataset['movieId'] == movie_id].index[0]
    distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)
    
    # Sort and get the nearest neighbors
    rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[1:]
    recommend_frame = []
    for val in rec_movie_indices:
        movie_idx = final_dataset.iloc[val[0]]['movieId']
        idx = movies[movies['movieId'] == movie_idx].index
        recommend_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
    
    df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend + 1))
    return df

print(get_movie_recommendation('Iron Man'))
print(get_movie_recommendation('Toy Story'))

