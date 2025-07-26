import pandas as pd 
import numpy as np
import joblib as jb
import warnings

# Function to extract unwatched movies for a specific user
def unwatched(userId, ratings_df, cleaned_df):
    all_movies = set(cleaned_df["id"])
    watchedMovies = set(ratings_df[ratings_df["userId"] == userId]["movieId"])
    unwatchedMovies = all_movies - watchedMovies

    # Compute user's mean stats
    total_rating_by_user = cleaned_df[cleaned_df['userId'] == userId]["total_rating_by_user"].mean()
    avg_rating_by_user = cleaned_df[cleaned_df['userId'] == userId]["avg_rating_by_user"].mean()

    # Get data for movies the user hasn't rated yet
    dummy = cleaned_df[cleaned_df["id"].isin(unwatchedMovies)].copy()

    # Drop duplicates and keep more popular movies first
    dummy = dummy.sort_values(by="total_rating_to_movie", ascending=False).drop_duplicates(subset="id")

    # Inject user-based features back in
    dummy["total_rating_by_user"] = total_rating_by_user
    dummy["avg_rating_by_user"] = avg_rating_by_user
    dummy["rating_deviation"] = dummy["avg_rating_by_user"] - dummy["avg_rating_to_movie"]

    # Keep a copy of the movie IDs for later mapping
    movie_ids = dummy["id"].copy()

    # Drop identifiers and prepare for model input
    dummy.drop(columns=["userId", "id"], inplace=True)

    return dummy, movie_ids

# Load saved model and cleaned data
model = jb.load("model.pkl")
x = jb.load("cleaned_data.pkl")
title_dict = jb.load("title.pkl")  # Dictionary: {movie_id: title}

# Drop label column before making predictions
x.drop("rating", axis=1, inplace=True)

# Load ratings (only required columns)
ratings = pd.read_csv("ratings.csv", usecols=["userId", "movieId", "rating"])

# Select a user for recommendation
userId_specific = 10

# Get feature data for unseen movies
dummy, movie_ids = unwatched(userId_specific, ratings, x)
X_test = dummy.values

# Suppress XGBoost's device mismatch warning during prediction
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    predictions = model.predict(X_test)

# Append predictions and restore IDs
dummy["predicted_rating"] = predictions
dummy["id"] = movie_ids.values

# Sort by predicted rating and select top 10 movies
top10 = dummy.sort_values(by="predicted_rating", ascending=False).head(10)

# Print movie titles using title_dict mapping
print("Top 10 Recommended Movies:")
for movie_id in top10["id"]:
    title = title_dict.get(movie_id, "Unknown Title")
    print(title)
