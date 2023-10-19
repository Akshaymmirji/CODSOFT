import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the MovieLens dataset (you can download it from MovieLens: https://grouplens.org/datasets/movielens/)
# We assume you have two CSV files: 'movies.csv' and 'ratings.csv'

# Load movie data
movies = pd.read_csv('movies.csv')

# Load ratings data
ratings = pd.read_csv('ratings.csv')

# Merge the data to get movie ratings along with movie information
movie_data = pd.merge(ratings, movies, on='movieId')

# Pivot the data to create a user-item matrix
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating')

# Fill missing values with 0 (unrated movies)
user_movie_matrix = user_movie_matrix.fillna(0)

# Calculate the cosine similarity between movies
movie_similarity = cosine_similarity(user_movie_matrix.T)

# Create a DataFrame for the movie similarity matrix
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Function to get movie recommendations based on user preferences
def get_movie_recommendations(movie_name, user_rating):
    similar_scores = movie_similarity_df[movie_name] * user_rating
    similar_movies = similar_scores.sort_values(ascending=False)
    return similar_movies

# Sample user preferences
user_preferences = [
    ("Forrest Gump (1994)", 5),
    ("Shawshank Redemption, The (1994)", 4),
    ("Pulp Fiction (1994)", 5),
    ("Matrix, The (1999)", 3),
]

# Create an empty DataFrame to store recommendations
recommendations = pd.DataFrame()

# Generate movie recommendations based on user preferences
for movie, rating in user_preferences:
    recommendations = recommendations.append(get_movie_recommendations(movie, rating))

# Group and sum the recommendations to get the final scores
final_recommendations = recommendations.groupby(recommendations.index).sum()

# Sort the recommendations by score in descending order
final_recommendations = final_recommendations.sort_values(ascending=False)

# Filter out movies the user has already rated
user_rated_movies = user_movie_matrix.loc[1]
final_recommendations = final_recommendations.drop(user_rated_movies[user_rated_movies > 0].index)

# Display the top N movie recommendations
top_n = 10
print(f"Top {top_n} Movie Recommendations for the User:")
print(final_recommendations.head(top_n))
