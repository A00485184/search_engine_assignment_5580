"""
Movie Recommendation Engine
-------------------------

What the code does:
- Loads the MovieLens 100k 'u.item' dataset containing movie metadata (titles, genres).
- Accepts a set of up to 5 movie titles from the user via the command line.
- Uses a Content-Based Filtering approach computing Cosine Similarity on movie genres to find similar movies.
- Recommends up to 10 additional movies based on the user's input.
- Provides a brief explanation for each recommendation referencing the overlapping genres with the input movie.

What the code does not do:
- It does not use Collaborative Filtering (user ratings are ignored; the u.data file is not required).
- It does not recommend based on directors, actors, or descriptions, as that data is not in the genre vector.
- It does not handle complex typos or misspellings perfectly, relying instead on simple case-insensitive substring matching for titles.

Specific algorithm used:
- Content-Based Filtering using Cosine Similarity on the 19 binary genre features per movie.

Assumptions made:
- Assumes users who like a specific genre will enjoy other movies with a highly similar genre profile.
- Assumes the 'u.item' file is correctly formatted with pipe separators and Latin-1 encoding, per the standard MovieLens 100k format.
- Assumes that if a user searches for a movie without specifying the year, returning the first matched movie containing that substring is acceptable.
- Assumes tie-breaking in similarity scores can be arbitrarily resolved via simple sorting sequence based on their original dataset index.

Instructions for setup:
1. Ensure you have python installed.
2. Install the required libraries using: `pip install pandas scikit-learn numpy`
3. Download the MovieLens 100k dataset from GroupLens (https://grouplens.org/datasets/movielens/100k/)
4. Extract the dataset and place the 'u.item' file in the same directory as this script.
5. Run the script from the terminal: `python movie_recommendation.py`
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Define the genre columns as present in the MovieLens 100k generic schema
GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", 
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# Total columns layout of the u.item pipe-separated list
COLUMNS = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url'] + GENRE_COLUMNS

def load_movie_data(file_path='u.item'):
    """
    Loads moving data from the u.item file.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found in the current directory.")
        print("Please ensure you have downloaded the MovieLens 100k dataset and 'u.item' is available.")
        return None
        
    try:
        df = pd.read_csv(file_path, sep='|', encoding='latin-1', names=COLUMNS, usecols=range(24))
        return df
    except Exception as e:
        print(f"Error parsing the data file: {e}")
        return None

def find_movie_by_title(title, movies_df):
    """
    Finds a movie in the dataframe using case-insensitive substring matching.
    Returns the first matching row, or None if not found.
    """
    matches = movies_df[movies_df['movie_title'].str.contains(title, case=False, na=False)]
    if not matches.empty:
        return matches.iloc[0]
    return None

def get_genres(movie_row):
    """Returns a list of genre names that are active (1) for a given movie row."""
    return [genre for genre in GENRE_COLUMNS if movie_row[genre] == 1]

def recommend_movies(input_movies, movies_df, top_n=10):
    """
    Generates up to `top_n` movie recommendations based on Cosine Similarity of genre vectors.
    """
    # Extract the genre matrix for all movies in the dataset
    genre_matrix = movies_df[GENRE_COLUMNS].values
    
    recommendations = []
    
    # Analyze similarities specifically corresponding to each user input movie to provide explanations
    for input_movie in input_movies:
        input_idx = input_movie.name  # The integer index in the dataframe
        target_genre_vector = input_movie[GENRE_COLUMNS].values.reshape(1, -1)
        
        # Calculate cosine similarity between this single input movie and all other movies
        similarities = cosine_similarity(target_genre_vector, genre_matrix)[0]
        
        # Don't recommend the exact same movie to itself
        similarities[input_idx] = -1.0 
        
        # Identify indices of the 20 most similar movies for this input item
        top_indices = np.argsort(similarities)[::-1][:20]
        
        input_genres = set(get_genres(input_movie))
        
        for idx in top_indices:
            sim_score = similarities[idx]
            if sim_score <= 0:
                continue
                
            rec_movie = movies_df.iloc[idx]
            rec_genres = set(get_genres(rec_movie))
            
            # Formulate the explanation based on the intersecting genres
            shared_genres = input_genres.intersection(rec_genres)
            if shared_genres:
                explanation = f"Because you liked '{input_movie['movie_title']}', which shares the same genre(s): {', '.join(shared_genres)}"
                
                recommendations.append({
                    'movie_id': rec_movie['movie_id'],
                    'movie_title': rec_movie['movie_title'],
                    'similarity': sim_score,
                    'explanation': explanation
                })
            
    # Sort the assembled aggregated pool of potential recommendations by similarity score descending
    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Filter final output: remove movies explicitly input by the user, and ensure variety by removing duplicates
    input_ids = {m['movie_id'] for m in input_movies}
    seen_ids = set()
    final_recommendations = []
    
    for rec in recommendations:
        rec_id = rec['movie_id']
        if rec_id not in input_ids and rec_id not in seen_ids:
            final_recommendations.append(rec)
            seen_ids.add(rec_id)
            
        if len(final_recommendations) >= top_n:
            break
            
    return final_recommendations

def main():
    print("Initializing Movie Recommendation Engine...")
    movies_df = load_movie_data('u.item')
    
    if movies_df is None:
        return
        
    print("\nWelcome to the Content-Based Movie Recommendation Engine!")
    print("Provide up to 5 movie titles you like. The algorithm will suggest 10 additional movies.")
    print("(Type 'done' if you want to finish entering movies early)\n")
    
    user_inputs = []
    while len(user_inputs) < 5:
        user_input = input(f"Enter movie #{len(user_inputs) + 1}: ").strip()
        
        if user_input.lower() == 'done':
            break
            
        if not user_input:
            continue
            
        movie = find_movie_by_title(user_input, movies_df)
        if movie is not None:
            print(f"  --> Found matched title: {movie['movie_title']}")
            
            # Check if this movie was already added
            if any(m['movie_id'] == movie['movie_id'] for m in user_inputs):
                print("  --> You already added this movie. Skipping.")
                continue
                
            user_inputs.append(movie)
        else:
            print(f"\n  Sorry, could not find any movie matching '{user_input}'. Please try again.\n")
            
    if not user_inputs:
        print("\nNo movies were appropriately entered. Exiting the engine.")
        return
        
    print(f"\nAnalyzing your preferences for {len(user_inputs)} movies based on Cosine Similarity...")
    recs = recommend_movies(user_inputs, movies_df, top_n=10)
    
    print("\n" + "="*70)
    print("                 TOP 10 MOVIE RECOMMENDATIONS")
    print("="*70 + "\n")
    
    if not recs:
        print("Could not find any suitable recommendations.")
    else:
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec['movie_title']}")
            print(f"   Explanation: {rec['explanation']}\n")
            
    print("="*70)

if __name__ == "__main__":
    main()
