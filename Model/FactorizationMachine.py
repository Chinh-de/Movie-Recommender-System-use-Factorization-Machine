import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import traceback


class FactorizationMachine(pl.LightningModule):
    def __init__(self, num_inputs, embedding_dim, lr=5e-4, weight_decay=5e-5, dropout_rate=0.2):
        super(FactorizationMachine, self).__init__()
        self.save_hyperparameters()
        
        self.rating_mean = (10 + 1) / 2
        self.rating_range = (10 - 1) / 2
        
        # Ma trận V
        self.embedding = nn.Embedding(num_inputs + 1, embedding_dim, padding_idx= num_inputs)
        
        # bias w
        self.linear = nn.Embedding(num_inputs + 1, 1, padding_idx= num_inputs)
        
        # Global bias w0
        self.bias = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(p=dropout_rate)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_normal_(self.embedding.weight.data, gain=1.0)
        nn.init.xavier_normal_(self.linear.weight.data, gain=1.0)
        nn.init.normal_(self.bias.data, mean=0.0, std=0.01)
        
    def forward(self, x):
        mask = (x != self.embedding.padding_idx).unsqueeze(-1)

        # Bias cho từng feature: w_i * x_i
        linear_part = self.linear(x).sum(dim=1).squeeze(-1)  # Thêm squeeze(-1) để chuyển từ [batch, 1] -> [batch]  

        # Ma trận V
        embed_x = self.embedding(x)  
        
        # Tính theo công thức  
        sum_of_embed = embed_x.sum(dim=1) 
        square_of_sum = sum_of_embed.pow(2) 
        
        sum_of_square = embed_x.pow(2).sum(dim=1) 
        
        # Phần cuối trong công thức
        interactions = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)
        
        # Tổng tất cả
        predictions = self.bias + linear_part + interactions
        
        return predictions
    
    def predict_rating(self, username, movie_id, user_index_by_username, movie_index_by_id, movie_features, user_features):

        self.eval()  
        with torch.no_grad():
            user_index = user_index_by_username.get(username, 0)
            movie_idx = movie_index_by_id.get(movie_id, 0)

            features = user_features[user_index] + movie_features[movie_idx]
            features_tensor = torch.tensor([features], dtype=torch.long)

            normalized_pred = self(features_tensor)
            original_pred = normalized_pred * self.rating_range + self.rating_mean
            return torch.clip(original_pred, min=1, max=10).item()
    
    def recommend_top_k(self, username, movie_features, user_features, user_index_by_username, k=10, exclude_movie_indices=None):

        self.eval() 
        if exclude_movie_indices is None:
            exclude_movie_indices = set()
            
        predictions = []
        
        with torch.no_grad():
            user_index = user_index_by_username.get(username, 0)
            
            for movie_idx in range(len(movie_features)):
                if movie_idx in exclude_movie_indices:
                    continue
                
                features = user_features[user_index] + movie_features[movie_idx]
                features_tensor = torch.tensor([features], dtype=torch.long)
                
                normalized_pred = self(features_tensor)
                original_pred = normalized_pred * self.rating_range + self.rating_mean
                
                predictions.append((movie_idx, original_pred.item()))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:k]
    
   
     
    def training_step(self, batch, batch_idx):
        inputs, ratings = batch
        
        # Normalize ratings to centered range
        normalized_ratings = (ratings - self.rating_mean) / self.rating_range
        
        predicted_ratings = self(inputs)
        loss = F.mse_loss(predicted_ratings, normalized_ratings)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, ratings = batch
        
        # Normalize ratings to centered range
        normalized_ratings = (ratings - self.rating_mean) / self.rating_range
        
        predicted_ratings = self(inputs)
        loss = F.mse_loss(predicted_ratings, normalized_ratings)
        
        # Calculate RMSE on original scale
        pred_original = predicted_ratings * self.rating_range + self.rating_mean
        rmse = torch.sqrt(F.mse_loss(pred_original, ratings))
        
        self.log("val_loss", loss)
        self.log("val_rmse", rmse)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )

    def on_validation_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_rmse = self.trainer.callback_metrics.get("val_rmse")
        
        train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        val_rmse_str = f"{val_rmse:.4f}" if val_rmse is not None else "N/A"
        
        print(f"Epoch {self.current_epoch}: Train Loss: {train_loss_str}, Val Loss: {val_loss_str}, Val RMSE: {val_rmse_str}")

    def find_potential_users(self, movie_id, movie_features, user_features, movie_index_by_id, username_by_index, k=10, exclude_user_indices=None):
        self.eval()
        
        # Khởi tạo set rỗng nếu exclude_user_indices là None  
        if exclude_user_indices is None:
            exclude_user_indices = set()

        print("find_potential_users debug 1")
        predictions = []

        with torch.no_grad():
            movie_idx = movie_index_by_id.get(movie_id, 0)
            movie_feature = movie_features[movie_idx]
            print("find_potential_users debug 2")
            # Duyệt qua từng user
            try:
                for user_idx in range(len(user_features)):
                    # Bỏ qua nếu user đã đánh giá
                    if user_idx in exclude_user_indices:
                        continue
                        
                    features = user_features[user_idx] + movie_feature
                    features_tensor = torch.tensor([features], dtype=torch.long)
                    
                    normalized_pred = self(features_tensor)
                    original_pred = normalized_pred * self.rating_range + self.rating_mean
                    
                    rounded_pred = round(original_pred.item(), 2)
                    predictions.append((username_by_index[user_idx], rounded_pred))
            except Exception as e:  
                print(f"{e}")
                print(traceback.format_exc())
                return []    
            
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:k]


# def create_user_features(username, gender, age, occupation, data):
#     """Tạo user features vector cho user cũ hoặc mới"""
#     features = []
    
#     # Thêm user index nếu là user cũ
#     if username in data["user_index_by_username"]:
#         features.append(data["user_index_by_username"][username])
    
#     # Thêm các feature khác
#     gender_index = data["gender_index_by_name"].get(gender)
#     if gender_index is not None:
#         features.append(gender_index + data["gender_offset"])
        
#     age_index = data["age_index_by_name"].get(age) 
#     if age_index is not None:
#         features.append(age_index + data["age_offset"])
        
#     occupation_index = data["occupation_index_by_name"].get(occupation)
#     if occupation_index is not None:
#         features.append(occupation_index + data["occupation_offset"])
        
#     return features

# def create_movie_features(movie_id, genres, year_group, runtime_group, directors, writers, stars, data):
#     """Tạo movie features vector cho movie cũ hoặc mới"""
#     features = []
    
#     # Thêm movie index nếu là movie cũ
#     if movie_id in data["movie_index_by_id"]:
#         features.append(data["movie_index_by_id"][movie_id] + data["movie_offset"])
    
#     # Thêm genres
#     if genres:
#         for genre in genres:
#             idx = data["genre_index_by_name"].get(genre.lower())
#             if idx is not None:
#                 features.append(idx + data["genre_offset"])
                
#     # Thêm year group
#     if year_group in data["year_index_by_group"]:
#         features.append(data["year_index_by_group"][year_group] + data["year_offset"])
        
#     # Thêm runtime group
#     if runtime_group in data["runtime_index_by_group"]:
#         features.append(data["runtime_index_by_group"][runtime_group] + data["runtime_offset"])
        
#     # Thêm directors
#     if directors:
#         for director in directors:
#             idx = data["director_index_by_name"].get(director.lower())
#             if idx is not None:
#                 features.append(idx + data["director_offset"])
                
#     # Thêm writers
#     if writers:
#         for writer in writers:
#             idx = data["writer_index_by_name"].get(writer.lower())
#             if idx is not None:
#                 features.append(idx + data["writer_offset"])
                
#     # Thêm stars
#     if stars:
#         for star in stars:
#             idx = data["star_index_by_name"].get(star.lower())
#             if idx is not None:
#                 features.append(idx + data["star_offset"])
                
#     return features

# def predict_rating(model, user_features, movie_features):
#     """Dự đoán rating cho một cặp user-movie"""
#     model.eval()
#     with torch.no_grad():
#         features = user_features + movie_features
#         features_tensor = torch.tensor([features], dtype=torch.long)
#         normalized_pred = model(features_tensor)
#         rating = normalized_pred * 4.5 + 5.5
#         return torch.clip(rating, min=1, max=10).item()
# def recommend_top_k(model, username, movie_features_list, user_features, ratings_df, movie_id_by_index, k=10):
#     """
#     Gợi ý top-k movies cho một user, loại bỏ những phim đã đánh giá
    
#     Args:
#         model: Mô hình FM
#         username: Username cần gợi ý  
#         movie_features_list: List features của movies
#         user_features: Features của user
#         ratings_df: DataFrame chứa ratings
#         movie_id_by_index: Dict ánh xạ movie index -> id
#         k: Số lượng phim cần gợi ý
#     """
#     # Lấy set các movie_id user đã đánh giá
#     rated_movies = set(ratings_df[ratings_df['Username'] == username]['MovieID'])
    
#     predictions = []
#     model.eval()
    
#     with torch.no_grad():
#         for idx, movie_features in enumerate(movie_features_list):
#             movie_id = movie_id_by_index.get(idx)
#             if movie_id in rated_movies:
#                 continue
                
#             features = user_features + movie_features
#             features_tensor = torch.tensor([features], dtype=torch.long)
#             normalized_pred = model(features_tensor)
#             rating = normalized_pred * 4.5 + 5.5
#             predictions.append((movie_id, rating.item()))
            
#     return sorted(predictions, key=lambda x: x[1], reverse=True)[:k]

# def find_potential_users_for_movie(model, movie_id, movie_features, user_features_list, 
#                                  ratings_df, username_by_index, k=10, rating_threshold=8.0):
#     """
#     Tìm top-k users có khả năng đánh giá cao movie
#     """
#     # Lấy set users đã đánh giá movie này
#     rated_users = set(ratings_df[ratings_df['MovieID'] == movie_id]['Username'])
    
#     predictions = []
#     model.eval()
    
#     with torch.no_grad():
#         for user_idx, user_features in enumerate(user_features_list):
#             username = username_by_index.get(user_idx)
#             if username in rated_users:
#                 continue
                
#             features = user_features + movie_features
#             features_tensor = torch.tensor([features], dtype=torch.long) 
#             normalized_pred = model(features_tensor)
#             rating = normalized_pred * 4.5 + 5.5

#             if rating.item() >= rating_threshold:
#                 predictions.append((username, rating.item()))
                
#     return sorted(predictions, key=lambda x: x[1], reverse=True)[:k]