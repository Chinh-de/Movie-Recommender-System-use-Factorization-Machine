import torch
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from New_Model.FactorizationMachine import FactorizationMachine

# Load supporting data
with open("New_Model/support_data.pkl", "rb") as f:
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
model.load_state_dict(torch.load("New_Model/factorization_machine.pt", map_location="cpu"))
model.eval()

# Load data for ratings and movies
ratings_df = pd.read_csv('data/ratings.csv')
movies_df = pd.read_csv('data/movies.csv')
users_df = pd.read_csv('data/users.csv')

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
                user_features=data["user_features"],
                user_index_by_username=user_index_by_username,
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
        recommended_movies=recommended_movies,
        user_ratings=user_ratings,
        movies_df=movies_df
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

@app.route('/new_user_recommend', methods=['GET', 'POST'])
def new_user_recommend():
    recommended_movies = []
    user_info = {'username': '', 'gender': '', 'age': '', 'occupation': ''}
    show_result = False

    # Các lựa chọn cho form
    gender_choices = [('Male', 'Nam'), ('Female', 'Nữ')]
    age_choices = [
        (1, 'Dưới 18'), (18, '18-24'), (25, '25-34'), (35, '35-44'), (45, '45-49'), (50, '50-55'), (56, '56+')
    ]
    occupation_choices = [
        (0, 'Khác'), (1, 'Giáo dục'), (2, 'Nghệ sĩ'), (3, 'Hành chính'), (4, 'Sinh viên'), (5, 'Chăm sóc khách hàng'),
        (6, 'Y tế'), (7, 'Quản lý'), (8, 'Nông dân'), (9, 'Nội trợ'), (10, 'Học sinh'), (11, 'Luật sư'),
        (12, 'Lập trình viên'), (13, 'Nghỉ hưu'), (14, 'Kinh doanh'), (15, 'Khoa học'), (16, 'Tự kinh doanh'),
        (17, 'Kỹ sư'), (18, 'Thợ thủ công'), (19, 'Thất nghiệp'), (20, 'Nhà văn')
    ]

    if request.method == 'POST':
        user_info['username'] = request.form.get('username', '').strip()
        user_info['gender'] = request.form.get('gender', '')
        user_info['age'] = int(request.form.get('age', 1))
        user_info['occupation'] = int(request.form.get('occupation', 0))
        show_result = True

        # Tạo user_index_by_username tạm thời
        temp_user_index = 0
        gender_index_by_name = {"Female":0, "Male": 1}
        age_index_by_name = {1: 0, 18: 1, 25: 2, 35:3, 45: 4, 50: 5, 56:6}
        occupation_index_by_name = {i: i for i in range(21)}
        num_users = 1
        gender_offset = num_users
        age_offset = gender_offset + len(gender_index_by_name)
        occupation_offset = age_offset + len(age_index_by_name)

        # Tạo user_features cho user mới
        gender_index = gender_index_by_name[user_info['gender']] + gender_offset
        age_index = age_index_by_name[user_info['age']] + age_offset
        occupation_index = occupation_index_by_name[user_info['occupation']] + occupation_offset
        new_user_features = [temp_user_index, gender_index, age_index, occupation_index]

        # Gợi ý phim cho user mới
        recommendations = model.recommend_top_k(
            user_info['username'],
            movie_features,
            [new_user_features],  # user_features chỉ có 1 user
            {user_info['username']: 0},
            exclude_movie_indices=set(),
            k=10
        )
        for i, (movie_idx, rating) in enumerate(recommendations, 1):
            movie_id = movie_id_by_index.get(movie_idx, "Unknown Movie ID")
            movie_row = movies_df[movies_df['ID'] == movie_id]
            if not movie_row.empty:
                movie_title = movie_row['Title'].values[0]
                movie_rating = movie_row['Rating'].values[0]
                movie_image = movie_row['Image'].values[0] if 'Image' in movie_row.columns else 'default_image_url.jpg'
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
        'new_user_recommend.html',
        user_info=user_info,
        recommended_movies=recommended_movies,
        show_result=show_result,
        gender_choices=gender_choices,
        age_choices=age_choices,
        occupation_choices=occupation_choices
    )


@app.route('/api/search_users')
def search_users():
    query = request.args.get('query', '').lower()
    
    if not query or len(query) < 2:
        return jsonify([])
    
    # Tìm các người dùng có username chứa query
    filtered_users = users_df[users_df['Username'].str.lower().str.contains(query)]
    
    # Giới hạn kết quả (tối đa 20 người dùng)
    filtered_users = filtered_users.head(20)
    
    # Chuyển đổi thành danh sách các dictionary
    results = []
    for _, row in filtered_users.iterrows():
        results.append({
            'username': row['Username'],
            'gender': row['Gender'],
            'age': int(row['Age']),
            'occupation': row['Occupation']
        })
    
    return jsonify(results)

@app.route('/add_movie', methods=['GET', 'POST'])
def add_movie():
    message = None
    predicted_users = []
    movie_info = {
        'ID': '', 'Title': '', 'Year': '', 'Genres': '', 'Image': '', 'Numrate': '', 'Rating': '',
        'Directors': '', 'Writers': '', 'Stars': '', 'Runtime': ''
    }
    if request.method == 'POST':
        # Lấy thông tin phim từ form
        for key in movie_info.keys():
            movie_info[key] = request.form.get(key, '').strip()
        # Thêm phim vào DataFrame và lưu file
        new_row = pd.DataFrame([movie_info])
        global movies_df
        movies_df = pd.concat([movies_df, new_row], ignore_index=True)
        movies_df.to_csv('data/movies.csv', index=False)
        message = f"Đã thêm phim '{movie_info['Title']}' thành công!"

        # Dự đoán điểm cho từng user
        top_users = []
        # --- Tạo movie_features cho phim mới từ dữ liệu nhập ---
        def get_or_default(val, default):
            return val if val else default

        # Lấy offset và mapping từ data
        genre_index_by_name = data.get('genre_index_by_name', {})
        year_index_by_group = data.get('year_index_by_group', {})
        runtime_index_by_group = data.get('runtime_index_by_group', {})
        director_index_by_name = data.get('director_index_by_name', {})
        writer_index_by_name = data.get('writer_index_by_name', {})
        star_index_by_name = data.get('star_index_by_name', {})
        occupation_offset = data.get('occupation_offset', 0)
        movie_offset = data.get('movie_offset', 0)
        genre_offset = data.get('genre_offset', 0)
        year_offset = data.get('year_offset', 0)
        runtime_offset = data.get('runtime_offset', 0)
        director_offset = data.get('director_offset', 0)
        writer_offset = data.get('writer_offset', 0)
        star_offset = data.get('star_offset', 0)

        # Tạo feature vector cho phim mới
        new_movie_features = []
        # Movie index mới
        new_movie_index = len(movies_df) - 1
        new_movie_features.append(movie_offset + new_movie_index)

        # Genres
        genres = [g.strip().lower() for g in movie_info['Genres'].split(',')] if movie_info['Genres'] else []
        if not genres and len(movie_features) > 0:
            # fallback
            genres = [g for g in data.get('sample_genres', ['other'])]
        for genre in genres:
            idx = genre_index_by_name.get(genre)
            if idx is not None:
                new_movie_features.append(genre_offset + idx)

        # Year group
        try:
            year = int(movie_info['Year'])
            year_group = (year // 5) * 5
        except:
            year_group = None
        if year_group in year_index_by_group:
            new_movie_features.append(year_offset + year_index_by_group[year_group])
        elif len(movie_features) > 0:
            # fallback
            sample = movie_features[0]
            for f in sample:
                if year_offset <= f < runtime_offset:
                    new_movie_features.append(f)

        # Runtime group
        try:
            runtime = int(movie_info['Runtime'])
            if runtime < 60:
                runtime_group = 'Very_Short'
            elif runtime < 90:
                runtime_group = 'Short'
            elif runtime < 120:
                runtime_group = 'Standard'
            elif runtime < 150:
                runtime_group = 'Long'
            else:
                runtime_group = 'Very_Long'
        except:
            runtime_group = None
        if runtime_group in runtime_index_by_group:
            new_movie_features.append(runtime_offset + runtime_index_by_group[runtime_group])
        elif len(movie_features) > 0:
            sample = movie_features[0]
            for f in sample:
                if runtime_offset <= f < director_offset:
                    new_movie_features.append(f)

        # Directors
        directors = [g.strip().lower() for g in movie_info['Directors'].split(',')] if movie_info['Directors'] else []
        if not directors and len(movie_features) > 0:
            directors = data.get('sample_directors', ['other'])
        for director in directors:
            idx = director_index_by_name.get(director)
            if idx is not None:
                new_movie_features.append(director_offset + idx)

        # Writers
        writers = [g.strip().lower() for g in movie_info['Writers'].split(',')] if movie_info['Writers'] else []
        if not writers and len(movie_features) > 0:
            writers = data.get('sample_writers', ['other'])
        for writer in writers:
            idx = writer_index_by_name.get(writer)
            if idx is not None:
                new_movie_features.append(writer_offset + idx)

        # Stars
        stars = [g.strip().lower() for g in movie_info['Stars'].split(',')] if movie_info['Stars'] else []
        if not stars and len(movie_features) > 0:
            stars = data.get('sample_stars', ['other'])
        for star in stars:
            idx = star_index_by_name.get(star)
            if idx is not None:
                new_movie_features.append(star_offset + idx)

        # Nếu thiếu feature, fallback lấy sample_movie_features
        if len(new_movie_features) < len(movie_features[0]):
            sample_movie_features = movie_features[0].copy()
            # Bổ sung cho đủ chiều dài
            while len(new_movie_features) < len(sample_movie_features):
                new_movie_features.append(sample_movie_features[len(new_movie_features)])

        # --- Dự đoán điểm cho từng user ---
        for username in ratings_df['Username'].unique():
            user_index = user_index_by_username.get(username, None)
            if user_index is None:
                continue
            user_features = data['user_features'][user_index]
            with torch.no_grad():
                features = user_features + new_movie_features
                features_tensor = torch.tensor([features], dtype=torch.long)
                pred = model(features_tensor)
                pred_score = float(pred.item())
            top_users.append((username, pred_score))
        predicted_users = sorted(top_users, key=lambda x: x[1], reverse=True)[:10]

    return render_template(
        'add_movie.html',
        message=message,
        movie_info=movie_info,
        predicted_users=predicted_users
    )

if __name__ == "__main__":
    app.run(debug=True)