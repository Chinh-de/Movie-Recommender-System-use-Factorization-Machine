import torch
import pickle
import pandas as pd
from Model.FactorizationMachine import FactorizationMachine

# ===== Load support data =====
with open("Model/support_data.pkl", "rb") as f:
    data = pickle.load(f)

# ===== Lấy thông số và dữ liệu =====
num_inputs = data["num_inputs"]
embedding_dim = data["embedding_dim"]
learning_rate = data["learning_rate"]
weight_decay = data["weight_decay"]
dropout_rate = data["dropout_rate"]

movie_features = data["movie_features"]
user_features = data["user_features"]
user_index_by_username = data["user_index_by_username"]
movie_index_by_id = data["movie_index_by_id"]
movie_id_by_index = {v: k for k, v in movie_index_by_id.items()}

# ===== Load model =====
model = FactorizationMachine(
    num_inputs=num_inputs,
    embedding_dim=embedding_dim,
    lr=learning_rate,
    weight_decay=weight_decay,
    dropout_rate=dropout_rate
)
model.load_state_dict(torch.load("Model/factorization_machine.pt", map_location="cpu"))
model.eval()

# ===== Predict test =====
username = "bretttaylor-04022"
movie_id = 'tt1654523'
user_index = user_index_by_username[username]
movie_index = movie_index_by_id[movie_id]


with torch.no_grad():
    prediction = model.predict_rating(
        username=username,
        movie_id=movie_id,
        user_index_by_username=user_index_by_username,
        movie_index_by_id=movie_index_by_id,
        movie_features=movie_features,
        user_features=user_features
    )
    print(f"⭐ Dự đoán rating của {username} cho phim {movie_id}: {prediction:.4f}")

# ===== Recommend top-k =====

ratings_df = pd.read_csv('data/ratings.csv')
movies_df = pd.read_csv('data/movies.csv')

movie_ids_set = set(ratings_df[ratings_df['Username'] == username]['MovieID'])
movie_idx_set = {movie_index_by_id.get(movie_id, 0) for movie_id in movie_ids_set}

    
recommendations = model.recommend_top_k(
    username,
    movie_features,
    user_index_by_username,
    k=10,
    exclude_movie_indices=movie_idx_set
)

print(f"Top 10 phim được gợi ý cho người dùng {username}:")
for i, (movie_idx, rating) in enumerate(recommendations, 1):
    movie_id = movie_id_by_index.get(movie_idx, "Unknown Movie ID")   
    movie_row = movies_df[movies_df['ID'] == movie_id]
    if not movie_row.empty:
        movie_title = movie_row['Title'].values[0]
        movie_rating = movie_row['Rating'].values[0]
    else:
        movie_title = "Unknown Movie Title"
        movie_rating = "Unknown Rating"
    print(f"{i}. {movie_title} (ID: {movie_id}) - Điểm dự đoán: {rating:.2f}/10 - Rating: {movie_rating}")
