import torch
import pickle
import pandas as pd
from flask import Flask, render_template, request
from Model.FactorizationMachine import FactorizationMachine

# Load supporting data
with open("Model/support_data.pkl", "rb") as f:
    data = pickle.load(f)

# Model parameters
num_inputs = data["num_inputs"]
embedding_dim = data["embedding_dim"]
learning_rate = data["learning_rate"]
weight_decay = data["weight_decay"]
dropout_rate = data["dropout_rate"]

movie_features = data["movie_features"]
user_index_by_username = data["user_index_by_username"]
movie_index_by_id = data["movie_index_by_id"]
movie_id_by_index = {v: k for k, v in movie_index_by_id.items()}

# Load trained model
model = FactorizationMachine(
    num_inputs=num_inputs,
    embedding_dim=embedding_dim,
    lr=learning_rate,
    weight_decay=weight_decay,
    dropout_rate=dropout_rate
)
model.load_state_dict(torch.load("Model/factorization_machine.pt", map_location="cpu"))
model.eval()

# Load data for ratings and movies
ratings_df = pd.read_csv('data/ratings.csv')
movies_df = pd.read_csv('data/movies.csv')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    user_ratings = None
    recommended_movies = []
    usernames = ratings_df['Username'].unique()

    if request.method == 'POST':
        # Updated to match form field name in the HTML template
        username = request.form['user_id']
        user_ratings = ratings_df[ratings_df['Username'] == username]

        if not user_ratings.empty:
            # Get the list of movie IDs that the user has already rated
            movie_ids_set = set(user_ratings['MovieID'])
            movie_idx_set = {movie_index_by_id.get(movie_id, 0) for movie_id in movie_ids_set}

            # Get top-k movie recommendations
            recommendations = model.recommend_top_k(
                username,
                movie_features,
                user_index_by_username,
                k=15,
                exclude_movie_indices=movie_idx_set
            )

            # Prepare recommended movies
            for i, (movie_idx, rating) in enumerate(recommendations, 1):
                movie_id = movie_id_by_index.get(movie_idx, "Unknown Movie ID")
                movie_row = movies_df[movies_df['ID'] == movie_id]

                if not movie_row.empty:
                    movie_title = movie_row['Title'].values[0]
                    movie_rating = movie_row['Rating'].values[0]
                    movie_image = movie_row['Image'].values[0] if 'Image' in movie_row.columns else 'default_image_url.jpg'
                    # Add the new fields
                    movie_year = movie_row['Year'].values[0] if 'Year' in movie_row.columns else 'Unknown'
                    movie_numrate = movie_row['Numrate'].values[0] if 'Numrate' in movie_row.columns else 'Unknown'
                    movie_genres = movie_row['Genres'].values[0] if 'Genres' in movie_row.columns else 'Unknown'
                else:
                    movie_title = "Unknown Movie Title"
                    movie_rating = "Unknown Rating"
                    movie_image = 'default_image_url.jpg'
                    movie_year = 'Unknown'
                    movie_numrate = 'Unknown'
                    movie_genres = 'Unknown'

                recommended_movies.append({
                    'title': movie_title,
                    'rating': movie_rating,
                    'movie_id': movie_id,
                    'predicted_rating': f"{rating:.2f}",
                    'image': movie_image,
                    'year': movie_year,
                    'numrate': movie_numrate,
                    'genres': movie_genres
                })

    return render_template(
        'index.html',
        usernames=usernames,
        user_ratings=user_ratings,
        recommended_movies=recommended_movies
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predicted_rating = None
    selected_username = None
    selected_movie_id = None
    movie_image = None
    avg_rating = None
    movie_title = None
    movie_year = None
    movie_numrate = None
    movie_genres = None
    usernames = ratings_df['Username'].unique()
    
    # Create a list of movie dictionaries for the dropdown
    movies_list = []
    for _, row in movies_df.iterrows():
        movies_list.append({
            'id': row['ID'],
            'title': row['Title']
        })

    if request.method == 'POST':
        # Updated to match form field names in the HTML template
        selected_username = request.form['user_id']
        selected_movie_id = request.form['movie_id']

        user_index = user_index_by_username.get(selected_username, -1)
        movie_index = movie_index_by_id.get(selected_movie_id, -1)

        if user_index != -1 and movie_index != -1:
            with torch.no_grad():
                predicted_rating_value = model.predict_rating(
                    username=selected_username,
                    movie_id=selected_movie_id,
                    user_index_by_username=user_index_by_username,
                    movie_index_by_id=movie_index_by_id,
                    movie_features=movie_features
                )
                predicted_rating = f"{predicted_rating_value:.2f}"

                # Get movie info
                movie_row = movies_df[movies_df['ID'] == selected_movie_id]
                if not movie_row.empty:
                    movie_image = movie_row['Image'].values[0] if 'Image' in movie_row.columns else 'default_image_url.jpg'
                    avg_rating = movie_row['Rating'].values[0]
                    movie_title = movie_row['Title'].values[0]
                    # Add the new fields
                    movie_year = movie_row['Year'].values[0] if 'Year' in movie_row.columns else 'Unknown'
                    movie_numrate = movie_row['Numrate'].values[0] if 'Numrate' in movie_row.columns else 'Unknown'
                    movie_genres = movie_row['Genres'].values[0] if 'Genres' in movie_row.columns else 'Unknown'
                else:
                    movie_image = 'default_image_url.jpg'
                    avg_rating = "Unknown Rating"
                    movie_title = "Unknown Movie"
                    movie_year = 'Unknown'
                    movie_numrate = 'Unknown'
                    movie_genres = 'Unknown'

    return render_template(
        'predict.html',
        predicted_rating=predicted_rating,
        selected_username=selected_username,
        selected_movie_id=selected_movie_id,
        movie_image=movie_image,
        avg_rating=avg_rating,
        movie_title=movie_title,
        movie_year=movie_year,
        movie_numrate=movie_numrate,
        movie_genres=movie_genres,
        usernames=usernames,
        movies_df=movies_list
    )

if __name__ == "__main__":
    app.run(debug=True)