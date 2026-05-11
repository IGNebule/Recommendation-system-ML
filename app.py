import pandas as pd
import re
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# INIT APP
# =========================
app = FastAPI()

# =========================
# CLEANING FUNCTION
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^a-z0-9]', ' ', text.lower()).strip()

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('../data/processed/games_content.csv')

# Normalize names ONCE (important)
df['name_clean'] = df['name'].apply(clean_text)

# =========================
# TF-IDF
# =========================
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(df['content'])

# =========================
# REQUEST MODEL
# =========================
class RecommendRequest(BaseModel):
    game_name: str
    top_n: int = 5

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(game_name, top_n=5):
    game_name = clean_text(game_name)

    print("\n=== DEBUG ===")
    print("INPUT:", game_name)

    # 🔥 Step 1: Exact match
    matches = df[df['name_clean'] == game_name]

    # 🔥 Step 2: Fallback (partial match)
    if matches.empty:
        matches = df[df['name_clean'].str.contains(game_name, na=False)]

    print("MATCH COUNT:", len(matches))
    print("MATCH SAMPLE:", matches['name'].head())

    if matches.empty:
        return []

    idx = matches.index[0]

    # 🔥 Efficient cosine similarity
    cosine_sim = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Get top scores
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]

    game_indices = [i[0] for i in sim_scores]

    results = []

    for i, score in sim_scores:
        results.append({
            "game": df['name'].iloc[i],
            "score": float(score)
        })

    print("RECOMMENDATIONS:", results)

    return results

# =========================
# API ROUTE
# =========================
@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    results = recommend(req.game_name, req.top_n)

    return {
        "input": req.game_name,
        "recommendations": results
    }