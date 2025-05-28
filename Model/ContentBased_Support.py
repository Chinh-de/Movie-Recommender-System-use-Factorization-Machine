import pickle
import pandas as pd
import numpy as np
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity
import ast
def add_prefix(list_items, prefix):
    if isinstance(list_items, str):
        
        try:
            list_items = ast.literal_eval(list_items)
        except:
            items = [item.strip() for item in list_items.split(',')]
            items = [item for item in items if item]  # bỏ phần tử rỗng
            return ' '.join([f"{prefix}{clean_string(item)}" for item in items if clean_string(item)])
    if not isinstance(list_items, list):
        return ''
    items = [item.strip() for item in list_items if item.strip()]  # loại bỏ rỗng
    return ' '.join([f"{prefix}{clean_string(item)}" for item in items if clean_string(item)])

def clean_string(s):
    s = s.replace(' ', '_')
    allowed_chars = 'abcdefghijklmnopqrstuvwxyz0123456789_'
    result = ''
    for ch in s:
        if ch in allowed_chars:
            result += ch
    return result

def get_similar_movies(movie_id, cosine_sim, movie_ids_list, movie_indices, top_n=10):
    if movie_id not in movie_indices:
        print("Movie ID không tồn tại.")
        return []

    idx = movie_indices[movie_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    similar_ids = [movie_ids_list[i[0]] for i in sim_scores]
    return similar_ids


def get_similar_movies(movie_id, cosine_sim, movie_ids_list, movie_indices, top_n=10):
    if movie_id not in movie_indices:
        print("Movie ID không tồn tại.")
        return []

    idx = movie_indices[movie_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    similar_ids = [movie_ids_list[i[0]] for i in sim_scores]
    return similar_ids

def combine_features(row):
    return ' '.join([
        add_prefix(row['Genres'], 'genre_'),
        add_prefix(row['Directors'], 'director_'),
        add_prefix(row['Writers'], 'writer_'),
        add_prefix(row['Stars'], 'star_'),
        f"runtime_{str(row['Runtime_group']).strip().lower()}",
        f"year_{str(row['Year']).strip().lower()}"
    ])


def add_new_movie(new_movie_row, vectorizer, tfidf_matrix, cosine_sim, movies_df, movie_indices):

    # Kiểm tra nếu phim mới đã tồn tại thì không thêm
    new_id = new_movie_row.iloc[0]['ID']

    # Transform features phim mới
    new_tfidf = vectorizer.transform(new_movie_row['combined_features'])

    # Nối tfidf_matrix cũ + mới
    tfidf_matrix = vstack([tfidf_matrix, new_tfidf])

    # Cập nhật movies_df
    movies_df = pd.concat([movies_df, new_movie_row], ignore_index=True)

    # Cập nhật movie_indices
    movie_indices = pd.Series(movies_df.index, index=movies_df['ID']).drop_duplicates()

    # Tính lại cosine similarity phần mới
    N_old = cosine_sim.shape[0]
    N_new = tfidf_matrix.shape[0]

    new_cosine_sim = np.zeros((N_new, N_new))

    # Copy cosine_sim cũ
    new_cosine_sim[:N_old, :N_old] = cosine_sim

    # similarity phim mới với phim cũ
    sim_new_old = cosine_similarity(new_tfidf, tfidf_matrix[:N_old])
    # similarity phim mới với phim mới
    sim_new_new = cosine_similarity(new_tfidf, new_tfidf)

    # Cập nhật ma trận
    new_cosine_sim[:N_old, N_old:] = sim_new_old.T
    new_cosine_sim[N_old:, :N_old] = sim_new_old
    new_cosine_sim[N_old:, N_old:] = sim_new_new

    cosine_sim = new_cosine_sim

    return vectorizer, tfidf_matrix, cosine_sim, movies_df, movie_indices

