import string
import pickle
import ast
import requests
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------
# Setup
# ----------------------------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

TMDB_KEY = "d5ccf3e7fe0bbb224765e797c0fef067"
POSTER_FALLBACK = "https://via.placeholder.com/500x750?text=No+Image"
PERSON_FALLBACK = "https://via.placeholder.com/300x400?text=No+Image"


# ----------------------------------
# API Helpers
# ----------------------------------
def safe_request(url):
    try:
        r = requests.get(url, timeout=3)  # Reduced timeout for speed
        if r.status_code == 200:
            return r.json()
        return None
    except requests.exceptions.RequestException:
        return None


def fetch_posters(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_KEY}"
    data = safe_request(url)
    if data and data.get("poster_path"):
        return "https://image.tmdb.org/t/p/w500" + data["poster_path"]
    return POSTER_FALLBACK


def fetch_person_details(person_id):
    url = f"https://api.themoviedb.org/3/person/{person_id}?api_key={TMDB_KEY}"
    data = safe_request(url)
    if not data:
        return PERSON_FALLBACK, "No biography available."
    profile = data.get("profile_path")
    bio = data.get("biography", "")
    image_url = ("https://image.tmdb.org/t/p/w500" + profile) if profile else PERSON_FALLBACK
    return image_url, bio


# ----------------------------------
# Parallel Helpers (New)
# ----------------------------------
def fetch_posters_parallel(movie_ids):
    """Fetch multiple posters in parallel."""
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_posters, movie_ids))
    return results


def fetch_cast_parallel(cast_ids):
    """Fetch multiple cast details in parallel."""
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_person_details, cast_ids))
    return results


# ----------------------------------
# Data Loading & Text Processing
# ----------------------------------
def get_list(obj):
    if isinstance(obj, list): return obj
    try:
        return ast.literal_eval(obj)
    except:
        return []


def stemming_stopwords(words_list):
    if not isinstance(words_list, list): return ""
    result = [ps.stem(w.lower()) for w in words_list if w.lower() not in stop_words and len(w) > 2]
    return " ".join(result).translate(str.maketrans("", "", string.punctuation))


# ----------------------------------
# Recommendation Logic (Optimized)
# ----------------------------------
def recommend_overall(new_df, movie, paths, weights=None, top_n=5, return_scores=False):
    if movie not in new_df["title"].values:
        return ([], []) if not return_scores else ([], [], [])

    if weights is None: weights = [1.0] * len(paths)

    df = new_df.reset_index(drop=True)
    idx = int(df[df["title"] == movie].index[0])
    n = df.shape[0]
    combined = np.zeros(n, dtype=float)

    # Load matrices
    for pth, w in zip(paths, weights):
        try:
            sim = pickle.load(open(pth, "rb"))
            sim_vec = np.array(sim[idx], dtype=float)

            # Normalize
            if idx < len(sim_vec): sim_vec[idx] = 0.0
            maxv, minv = sim_vec.max(), sim_vec.min()
            if maxv > 0 and maxv != minv:
                sim_vec = (sim_vec - minv) / (maxv - minv)

            combined += w * sim_vec
        except Exception as e:
            print(f"Skipping {pth}: {e}")
            continue

    combined[idx] = -1.0

    # Get Top N indices
    top_idx = np.argsort(combined)[::-1][:top_n]
    top_idx = [int(i) for i in top_idx if combined[int(i)] > -0.5]

    # Gather data
    movies_list = [df.iloc[i]["title"] for i in top_idx]
    movie_ids = [int(df.iloc[i]["movie_id"]) for i in top_idx]

    # PARALLEL FETCHING HERE
    posters = fetch_posters_parallel(movie_ids)

    if return_scores:
        scored = [(df.iloc[i]["title"], float(combined[i])) for i in top_idx]
        return movies_list, posters, scored

    return movies_list, posters


def get_details(name):
    try:
        movies = pd.DataFrame.from_dict(pickle.load(open("Files/movies_dict.pkl", "rb")))
        movies2 = pd.DataFrame.from_dict(pickle.load(open("Files/movies2_dict.pkl", "rb")))
    except FileNotFoundError:
        return None

    if name not in movies["title"].values:
        return None

    a = movies2[movies2["title"] == name].iloc[0]
    b = movies[movies["title"] == name].iloc[0]

    movie_id = a["movie_id"]
    poster = fetch_posters(movie_id)

    # Get raw cast IDs
    raw_cast = get_list(b["cast"])
    cast_ids = [c["id"] for c in raw_cast][:5]
    cast_names = [c["name"] for c in raw_cast][:5]

    langs = [d["name"] for d in get_list(a["spoken_languages"])]

    return {
        "poster": poster,
        "budget": a["budget"],
        "genres": b["genres"],
        "overview": a["overview"],
        "date": a["release_date"],
        "revenue": a["revenue"],
        "runtime": a["runtime"],
        "rating": a["vote_average"],
        "votes": a["vote_count"],
        "id": movie_id,
        "director": b["director"],
        "cast_ids": cast_ids,
        "cast_names": cast_names
    }