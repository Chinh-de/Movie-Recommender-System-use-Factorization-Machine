import torch
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for
from Model.FactorizationMachine import FactorizationMachine
from Model.ContentBased_Support import get_similar_movies, combine_features, add_new_movie
import time
from flask import redirect
# Load supporting data
with open("Model/support_data.pkl", "rb") as f:
    data = pickle.load(f)
with open('Model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('Model/tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)
with open('Model/cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# Model parameters
num_inputs = data["num_inputs"]
embedding_dim = data["embedding_dim"]
learning_rate = data["learning_rate"]
weight_decay = data["weight_decay"]
dropout_rate = data["dropout_rate"]

movie_features_list = data["movie_features"]
user_features_list = data["user_features"]

user_data = {
    "user_index_by_username": data["user_index_by_username"],
    "gender_index_by_name": data["gender_index_by_name"],
    "age_index_by_name": data["age_index_by_name"],
    "occupation_index_by_name": data["occupation_index_by_name"],
    "gender_offset": data["gender_offset"],
    "age_offset": data["age_offset"],
    "occupation_offset": data["occupation_offset"],
}

movie_data = {
    "movie_index_by_id": data["movie_index_by_id"],
    "genre_index_by_name": data["genre_index_by_name"],
    "year_index_by_group": data["year_index_by_group"],
    "runtime_index_by_group": data["runtime_index_by_group"],
    "director_index_by_name": data["director_index_by_name"],
    "writer_index_by_name": data["writer_index_by_name"],
    "star_index_by_name": data["star_index_by_name"],
    "movie_offset": data["movie_offset"],
    "genre_offset": data["genre_offset"],
    "year_offset": data["year_offset"],
    "runtime_offset": data["runtime_offset"],
    "director_offset": data["director_offset"],
    "writer_offset": data["writer_offset"],
    "star_offset": data["star_offset"],
}

movie_id_by_index = {v: k for k, v in movie_data["movie_index_by_id"].items()}
username_by_index = {v: k for k, v in user_data["user_index_by_username"].items()}

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
users_df = pd.read_csv('data/users.csv')
movie_indices = pd.Series(movies_df.index, index=movies_df['ID']).drop_duplicates()
movie_ids_list = movies_df['ID'].tolist()


app = Flask(__name__) 

@app.route('/', methods=['GET', 'POST'])
def index():
    user_ratings = None
    recommended_movies = []
    usernames = users_df['Username']

    if request.method == 'POST':
        # Updated to match form field name in the HTML template
        username = request.form['user_id']
        mask = ratings_df['Username'].eq(username)
        user_ratings = ratings_df.loc[mask]

        if user_ratings.empty:
            movie_idx_set = set()  # No ratings, so no exclusions
        else:    
            movie_ids_set = set(user_ratings['MovieID'])
            movie_idx_set = {movie_data['movie_index_by_id'].get(movie_id, 0) for movie_id in movie_ids_set}

        # Get top-k movie recommendations
        recommendations = model.recommend_top_k(
            username=username,
            movie_features=movie_features_list,
            user_features=user_features_list,
            user_index_by_username=user_data["user_index_by_username"],
            k=15,
            exclude_movie_indices=movie_idx_set
        )
        print(f"Recommendations for user '{username}': {recommendations}")

        # Prepare recommended movies
        for i, (movie_idx, rating) in enumerate(recommendations, 1):
            movie_id = movie_id_by_index.get(movie_idx, "Unknown Movie ID")
            movie_row = movies_df[movies_df['ID'] == movie_id]

            if not movie_row.empty:
                movie_id = movie_row['ID'].values[0]
                movie_title = movie_row['Title'].values[0]
                movie_rating = movie_row['Rating'].values[0]
                movie_image = movie_row['Image'].values[0] if 'Image' in movie_row.columns else 'default_image_url.jpg'
                # Add the new fields
                movie_year = movie_row['Year'].values[0] if 'Year' in movie_row.columns else 'Unknown'
                movie_numrate = movie_row['Numrate'].values[0] if 'Numrate' in movie_row.columns else 'Unknown'
                movie_genres = movie_row['Genres'].values[0] if 'Genres' in movie_row.columns else 'Unknown'
            else:
                movie_id = "#"
                movie_title = "Unknown Movie Title"
                movie_rating = "Unknown Rating"
                movie_image = 'default_image_url.jpg'
                movie_year = 'Unknown'
                movie_numrate = 'Unknown'
                movie_genres = 'Unknown'

            recommended_movies.append({
                'ID': movie_id,
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

        user_index = user_data['user_index_by_username'].get(selected_username, -1)
        movie_index = movie_data['movie_index_by_id'].get(selected_movie_id, -1)

        if user_index != -1 and movie_index != -1:
            with torch.no_grad():
                predicted_rating_value = model.predict_rating(
                    username=selected_username,
                    movie_id=selected_movie_id,
                    user_index_by_username=user_data['user_index_by_username'],
                    movie_index_by_id=movie_data['movie_index_by_id'],
                    movie_features=movie_features_list,
                    user_features=user_features_list,
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
    global user_data, movie_data, model, movie_features_list, user_features_list, movies_df, users_df, movie_id_by_index, username_by_index

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

        if user_info['username'] in user_data['user_index_by_username'].keys():
            return render_template('error.html', error="Username đã tồn tại.")
        print(user_info)

        idx = len(user_data['user_index_by_username'])
        user_data['user_index_by_username'][user_info['username']] = idx
        username_by_index[idx] = user_info['username']

        # Tạo user_features cho user mới
        gender_index = user_data['gender_index_by_name'][user_info['gender']] + user_data['gender_offset']
        age_index = user_data['age_index_by_name'][user_info['age']] + user_data['age_offset']
        occupation_index = user_data['occupation_index_by_name'][user_info['occupation']] + user_data['occupation_offset']
        new_user_features = [num_inputs, gender_index, age_index, occupation_index]
        # num_inputs là padding 
        user_features_list.append(new_user_features)
        
        new_row = pd.DataFrame([{
            'Username': user_info['username'],
            'Gender': user_info['gender'],
            'Age': user_info['age'],
            'Occupation': user_info['occupation']
        }])

        # Nối vào users_df
        users_df = pd.concat([users_df, new_row], ignore_index=True)


        # Gợi ý phim cho user mới
        recommendations = model.recommend_top_k(
            user_info['username'],
            movie_features_list,
            user_features_list, 
            {user_info['username']: idx},  # user_index_by_username chỉ có 1 user
            exclude_movie_indices=set(),
            k=15
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
                'ID': movie_id,
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

@app.route('/api/search_movies')
def search_movies():
    query = request.args.get('query', '').lower()
    
    if not query or len(query) < 2:
        return jsonify([])

    # Tìm các phim có tiêu đề chứa query
    filtered_movies = movies_df[movies_df['Title'].str.lower().str.contains(query)]

    # Giới hạn kết quả (tối đa 20 phim)
    filtered_movies = filtered_movies.head(20)

    # Chuyển đổi thành danh sách các dictionary
    results = []
    for _, row in filtered_movies.iterrows():
        results.append({
            'id': row['ID'],
            'title': row['Title'],
            'year': int(row['Year']),
            'genres': row['Genres'],
            'image': row['Image'],
            'numrate': row['Numrate'],
            'rating': row['Rating']
        })

    return jsonify(results)



@app.route('/movies')
def movies_index():
    query = request.args.get('q', '').strip()
    
    # Lọc phim theo query nếu có
    if query:
        filtered_movies = movies_df[movies_df['Title'].str.contains(query, case=False, na=False)]
    else:
        filtered_movies = movies_df

    # Lấy 40 phim đầu tiên
    movies = filtered_movies.head(40).to_dict('records')

    return render_template(
        'movies.html',
        movies=movies,
        query=query
    )

def group_years(year):
    return (year // 5) * 5

def group_runtime(runtime):
    if runtime < 60:
        return 'Very_Short'
    elif runtime < 90:
        return 'Short'
    elif runtime < 120:
        return 'Standard'
    elif runtime < 150:
        return 'Long'
    else:
        return 'Very_Long'

@app.route('/api/search_movie_fields')
def search_movie_fields():
    field = request.args.get('field', '')
    query = request.args.get('query', '').lower()
    
    if not query or len(query) < 2:
        return jsonify([])
        
    results = []
    if field == 'genres':
        terms = [name for name in movie_data['genre_index_by_name'].keys() if query in name.lower()]
        results = [{'name': term} for term in terms]
    elif field == 'directors':
        terms = [name for name in movie_data['director_index_by_name'].keys() if query in name.lower()]
        results = [{'name': term} for term in terms]
    elif field == 'writers':
        terms = [name for name in movie_data['writer_index_by_name'].keys() if query in name.lower()]
        results = [{'name': term} for term in terms]
    elif field == 'stars':
        terms = [name for name in movie_data['star_index_by_name'].keys() if query in name.lower()]
        results = [{'name': term} for term in terms]
    
    return jsonify(results[:20])

@app.route('/add_movie', methods=['GET', 'POST'])
def add_movie():
    global movie_features_list, user_features_list, movies_df, movie_data, movie_id_by_index, vectorizer, tfidf_matrix, cosine_sim, movies_df, movie_indices, movie_ids_list
    message = None
    predicted_users = []
    movie_info = {
        'ID': '', 'Title': '', 'Year': '', 'Genres': '', 'Image': '', 'Numrate': '', 'Rating': '',
        'Directors': '', 'Writers': '', 'Stars': '', 'Runtime': ''
    }
    
    
    if request.method == 'POST':
        # Get movie info from form
        for key in movie_info.keys():
            movie_info[key] = request.form.get(key, '').strip()
        
        try:
            if movie_info['ID'] in movie_data['movie_index_by_id']:
                message = "Phim đã tồn tại trong cơ sở dữ liệu."
                return render_template('add_movie.html', message=message, movie_info=movie_info, predicted_users=predicted_users)            
            # Convert string inputs to appropriate types and calculate groups
            movie_info['Year'] = int(movie_info['Year'])
            movie_info['Runtime'] = int(movie_info['Runtime'])
            movie_info['Numrate'] = 0  # New movies start with 0 ratings
            movie_info['Rating'] = 0.0  # New movies start with 0 rating
            
            # Calculate runtime group
            if movie_info['Runtime'] < 60:
                runtime_group = 'Very_Short'
            elif movie_info['Runtime'] < 90:
                runtime_group = 'Short'
            elif movie_info['Runtime'] < 120:
                runtime_group = 'Standard'
            elif movie_info['Runtime'] < 150:
                runtime_group = 'Long'
            else:
                runtime_group = 'Very_Long'
                
            # Store calculated runtime group
            movie_info['Runtime_group'] = runtime_group
                       
            
            # Create movie features for the new movie
            movie_features = [num_inputs]  # Start with padding
            movie_index = len(movie_data['movie_index_by_id'])
            movie_data['movie_index_by_id'][movie_info['ID']] = movie_index
            movie_id_by_index[movie_index] = movie_info['ID']


            # Add genres
            if movie_info['Genres']:
                for genre in movie_info['Genres'].split(', '):
                    if genre in movie_data['genre_index_by_name']:
                        idx = movie_data['genre_index_by_name'][genre]
                        movie_features.append(movie_data['genre_offset'] + idx)
                    
            # Add year group
            if movie_info['Year'] < 2000:
                year_group = 2000
            elif movie_info['Year'] > 2024:
                year_group = 2000
            else:
                year_group = (movie_info['Year'] // 5) * 5
            movie_info['Year_group'] = year_group
            if year_group in movie_data['year_index_by_group']:
                idx = movie_data['year_index_by_group'][year_group]
                movie_features.append(movie_data['year_offset'] + idx)
                  # Add runtime group - using calculated group
            if runtime_group in movie_data['runtime_index_by_group']:
                idx = movie_data['runtime_index_by_group'][runtime_group]
                movie_features.append(movie_data['runtime_offset'] + idx)
                
            # Add directors
            if movie_info['Directors']:
                for director in movie_info['Directors'].split(', '):
                    if director in movie_data['director_index_by_name']:
                        idx = movie_data['director_index_by_name'][director]
                        movie_features.append(movie_data['director_offset'] + idx)
                    
            # Add writers
            if movie_info['Writers']:
                for writer in movie_info['Writers'].split(', '):
                    if writer in movie_data['writer_index_by_name']:
                        idx = movie_data['writer_index_by_name'][writer]
                        movie_features.append(movie_data['writer_offset'] + idx)
                    
            # Add stars
            if movie_info['Stars']:
                for star in movie_info['Stars'].split(', '):
                    if star in movie_data['star_index_by_name']:
                        idx = movie_data['star_index_by_name'][star]
                        movie_features.append(movie_data['star_offset'] + idx)



            # Update global movie features list
            movie_features_list.append(movie_features)            

            new_row = pd.DataFrame([movie_info])
            new_row['combined_features'] = new_row.apply(combine_features, axis=1)

            vectorizer, tfidf_matrix, cosine_sim, movies_df, movie_indices = add_new_movie(new_row, vectorizer, tfidf_matrix, cosine_sim, movies_df, movie_indices)
            print(f"Added new movie: {new_row}")
            movie_ids_list.append(movie_info['ID'])
            return redirect(url_for('movie_detail', movieid=movie_info['ID']))
        except Exception as e:
            message = f"Lỗi khi thêm phim: {str(e)}"
            print(f"Error while adding movie: {e}")
            
    return render_template(
        'add_movie.html',
        message=message,
        movie_info=movie_info,
        predicted_users=predicted_users,

    )


def process_array_field(field_value):
    if isinstance(field_value, str):
        # Remove brackets and quotes, then split by comma
        clean_str = field_value.strip('[]').replace("'", "").strip()
        return ', '.join(item.strip() for item in clean_str.split(','))
    return field_value

@app.route('/movie/<movieid>')
def movie_detail(movieid):
    try:
        movie_row = movies_df[movies_df['ID'] == movieid]
        if movie_row.empty:
            return "Không tìm thấy phim", 404
            
        movie = movie_row.iloc[0].to_dict()

        # Process array fields
        array_fields = ['Directors', 'Writers', 'Stars', 'Genres']
        for field in array_fields:
            if field in movie:
                movie[field] = process_array_field(movie[field])


        similar_movie_ids = get_similar_movies(
            movie_id=movieid,
            cosine_sim=cosine_sim,
            movie_ids_list=movie_ids_list,
            movie_indices=movie_indices,
            top_n=10
        )
        similar_movies = []
        for m_id in similar_movie_ids:
            similar_movie = movies_df[movies_df['ID'] == m_id].iloc[0].to_dict()
            for field in array_fields:
                if field in similar_movie:
                    similar_movie[field] = process_array_field(similar_movie[field])
                if field == 'Genres' and len(similar_movie[field]) > 30:
                    genres = similar_movie[field].split(', ')
                    similar_movie[field] = ', '.join(genres[:2]) + '...'
            similar_movies.append(similar_movie)
        return render_template('movie_detail.html',
                            movie=movie,
                            similar_movies=similar_movies)
    except (IndexError, KeyError):
        return "Không tìm thấy phim", 404

@app.route('/api/potential_users/<movieid>')
def get_potential_users(movieid):
    try:  
        mask = ratings_df['MovieID'].eq(movieid)
        user_ratings = ratings_df.loc[mask]
        
        if user_ratings.empty:
            username_idx_set = None
        else:    
            usernames_set = set(user_ratings['Username'])
            username_idx_set = {user_data['user_index_by_username'].get(username, 0) for username in usernames_set}


        potential_users = model.find_potential_users(
            movie_id=movieid,
            movie_features=movie_features_list,
            user_features=user_features_list,
            username_by_index=username_by_index,
            movie_index_by_id=movie_data['movie_index_by_id'],
            k=10,
            exclude_user_indices=username_idx_set,
        )
        
        # Chuyển đổi điểm số sang thang 1-10
        potential_users = [
            (username, round(min(max(rating, 1.0), 10.0), 2))
            for username, rating in potential_users
        ]
        
        return jsonify(potential_users)
    except Exception as e:
        print(f"Error in get_potential_users: {e}")
        
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)