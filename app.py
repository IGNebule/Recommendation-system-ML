import pandas as pd
import numpy as np
import re
from collections import Counter
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

df = pd.read_csv('./data/games_content.csv')
df['appid'] = df['appid'].astype(str)

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(df['content'])

class RecommendRequest(BaseModel):
    appid: int
    top_n: int = 5

def recommend(appid, top_n=5):
    appid = str(appid)

    print("\n=== DEBUG ===")
    print("INPUT APPID:", appid)

    matches = df[df['appid'] == appid]

    print("MATCH COUNT:", len(matches))

    if matches.empty:
        return []

    idx = matches.index[0]

    cosine_sim = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    sim_scores = list(enumerate(cosine_sim))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n+1]

    results = []

    for i, score in sim_scores:
        results.append({
            "appid": df['appid'].iloc[i],
            "name": df['name'].iloc[i],
            "score": float(score)
        })

    print("RECOMMENDATIONS:", results)

    return results

# =========================
# API ROUTE
# =========================
@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    results = recommend(req.appid, req.top_n)

    return {
        "input_appid": req.appid,
        "recommendations": results
    }

class DebugRequest(BaseModel):
    text: str


def get_document_frequency_map():
    feature_names = tfidf.get_feature_names_out()

    df_counts = np.asarray(
        (tfidf_matrix > 0).sum(axis=0)
    ).ravel()

    return {
        feature_names[i]: int(df_counts[i])
        for i in range(len(feature_names))
    }


@app.get("/report/corpus")
def get_corpus_report():
    total_documents = int(df.shape[0])
    vocabulary_size = int(len(tfidf.vocabulary_))

    total_document_tokens = int(
        df["content"]
        .fillna("")
        .astype(str)
        .apply(lambda text: len(text.split()))
        .sum()
    )

    non_zero_elements = int(tfidf_matrix.nnz)
    rows, cols = tfidf_matrix.shape
    total_matrix_cells = int(rows * cols)

    matrix_sparsity = (
        1 - (non_zero_elements / total_matrix_cells)
        if total_matrix_cells > 0
        else 0
    )

    feature_names = tfidf.get_feature_names_out()
    document_frequency = get_document_frequency_map()

    target_terms = [
        "action",
        "indie",
        "rpg",
        "strategy",
        "roguelike",
    ]

    target_term_distribution = []

    for term in target_terms:
        df_value = document_frequency.get(term, 0)

        target_term_distribution.append({
            "term": term,
            "document_frequency": df_value,
            "percentage": round(
                (df_value / total_documents) * 100,
                2
            ) if total_documents > 0 else 0
        })

    top_terms = sorted(
        document_frequency.items(),
        key=lambda item: item[1],
        reverse=True
    )[:25]

    return {
        "source": "games_content.csv",
        "total_documents": total_documents,
        "vocabulary_size": vocabulary_size,
        "total_document_tokens": total_document_tokens,
        "tfidf_shape": [int(rows), int(cols)],
        "non_zero_elements": non_zero_elements,
        "total_matrix_cells": total_matrix_cells,
        "matrix_sparsity": matrix_sparsity,
        "matrix_sparsity_percent": round(matrix_sparsity * 100, 2),
        "vectorizer": {
            "type": "TfidfVectorizer",
            "stop_words": "english",
            "max_features": 5000
        },
        "target_term_distribution": target_term_distribution,
        "top_terms": [
            {
                "term": term,
                "document_frequency": count,
                "percentage": round((count / total_documents) * 100, 2)
                if total_documents > 0 else 0
            }
            for term, count in top_terms
        ]
    }


@app.post("/report/debug")
def debug_vector_pipeline(req: DebugRequest):
    raw_text = req.text or ""

    normalized = re.sub(r"[^a-z0-9\s-]", " ", raw_text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    stage_one_tokens = normalized.split() if normalized else []

    analyzer = tfidf.build_analyzer()
    stage_two_tokens = analyzer(raw_text)

    vector = tfidf.transform([raw_text])
    feature_names = tfidf.get_feature_names_out()

    weights = {}

    for index in vector.nonzero()[1]:
        token = feature_names[index]
        weight = vector[0, index]

        weights[token] = round(float(weight), 4)

    sorted_weights = dict(
        sorted(
            weights.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    return {
        "input": raw_text,
        "stage_1_tokenization_normalization": stage_one_tokens,
        "stage_2_stop_word_elimination": stage_two_tokens,
        "stage_3_tfidf_weights": sorted_weights
    }