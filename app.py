from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import os
import gdown

# ====== CONFIG ======
MODEL_PATH = "pytorch_movie_recommender.pt"
MOVIES_CSV = "movies.csv"
RATINGS_CSV = "ratings.csv"
GOOGLE_DRIVE_FILE_ID = "17vzRgPGhVHDCidRjrdNm0tJIPMkXawTR"  # <-- Replace this with your real file ID
# =====================

# Download ratings.csv if missing
def download_ratings():
    if not os.path.exists(RATINGS_CSV):
        print("Downloading ratings.csv from Google Drive...")
        url = f"https://drive.google.com/uc?id=17vzRgPGhVHDCidRjrdNm0tJIPMkXawTR"
        gdown.download(url, RATINGS_CSV, quiet=False)

# Setup
app = Flask(__name__)
CORS(app)

# Download ratings.csv
download_ratings()

# Load resources
print("Loading model and data...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model (same as during training)
class RecommenderNet(torch.nn.Module):
    def __init__(self, num_users, num_movies):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, 50)
        self.movie_embedding = torch.nn.Embedding(num_movies, 50)
        self.fc1 = torch.nn.Linear(100, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, user_ids, movie_ids):
        user_vec = self.user_embedding(user_ids)
        movie_vec = self.movie_embedding(movie_ids)
        x = torch.cat([user_vec, movie_vec], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x).squeeze()

# Prepare data
movies_df = pd.read_csv(MOVIES_CSV)
ratings_df = pd.read_csv(RATINGS_CSV)

user_ids = ratings_df['userId'].unique()
movie_ids = ratings_df['movieId'].unique()

user_to_index = {u: i for i, u in enumerate(user_ids)}
movie_to_index = {m: i for i, m in enumerate(movie_ids)}
index_to_movie = {i: m for m, i in movie_to_index.items()}

num_users = len(user_to_index)
num_movies = len(movie_to_index)

# Load model
model = RecommenderNet(num_users, num_movies).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# API Route
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)

    if user_id not in user_to_index:
        return jsonify({"error": "User ID not found."}), 404

    user_idx = user_to_index[user_id]
    all_movie_indices = np.array(list(movie_to_index.values()))

    user_tensor = torch.tensor([user_idx] * len(all_movie_indices), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(all_movie_indices, dtype=torch.long).to(device)

    with torch.no_grad():
        scores = model(user_tensor, movie_tensor).cpu().numpy()

    top_indices = np.argsort(scores)[-5:][::-1]  # Top 5
    recommended_movie_ids = [index_to_movie[i] for i in top_indices]
    recommended_titles = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]['title'].tolist()

    return jsonify({"recommended_movies": recommended_titles})

# Start server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
