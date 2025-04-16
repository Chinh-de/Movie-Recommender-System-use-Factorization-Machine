import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader


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
    
    def predict_rating(self, username, movie_id, user_index_by_username, movie_index_by_id, movie_features):

        self.eval()  
        with torch.no_grad():
            user_index = user_index_by_username.get(username, 0)
            movie_idx = movie_index_by_id.get(movie_id, 0)

            features = [user_index] + movie_features[movie_idx]
            features_tensor = torch.tensor([features], dtype=torch.long)

            normalized_pred = self(features_tensor)
            original_pred = normalized_pred * self.rating_range + self.rating_mean
            return torch.clip(original_pred, min=1, max=10).item()
    
    def recommend_top_k(self, username, movie_features, user_index_by_username, k=10, exclude_movie_indices=None):

        self.eval() 
        if exclude_movie_indices is None:
            exclude_movie_indices = set()
            
        predictions = []
        
        with torch.no_grad():
            user_index = user_index_by_username.get(username, 0)
            
            for movie_idx in range(len(movie_features)):
                if movie_idx in exclude_movie_indices:
                    continue
                
                features = [user_index] + movie_features[movie_idx]
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