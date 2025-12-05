import os
import pickle
import pandas as pd
from processing import preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Main:
    def __init__(self):
        self.new_df = None
        self.movies = None
        self.movies2 = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # ----------------------------
    # Load pickle or generate DF
    # ----------------------------
    def get_df(self):
        movies_pkl = "Files/movies_dict.pkl"
        movies2_pkl = "Files/movies2_dict.pkl"
        newdf_pkl = "Files/new_df_dict.pkl"

        if all(os.path.exists(p) for p in [movies_pkl, movies2_pkl, newdf_pkl]):
            try:
                self.movies = pd.DataFrame.from_dict(pickle.load(open(movies_pkl, 'rb')))
                self.movies2 = pd.DataFrame.from_dict(pickle.load(open(movies2_pkl, 'rb')))
                self.new_df = pd.DataFrame.from_dict(pickle.load(open(newdf_pkl, 'rb')))
                return
            except Exception as e:
                print(f"Error loading pickle files: {e}. Regenerating DataFrames.")

        # Generate DataFrames if pickle files missing/corrupt
        self.movies, self.new_df, self.movies2 = preprocess.read_csv_to_df()

        # Save pickles
        pickle.dump(self.movies.to_dict(), open(movies_pkl, "wb"))
        pickle.dump(self.movies2.to_dict(), open(movies2_pkl, "wb"))
        pickle.dump(self.new_df.to_dict(), open(newdf_pkl, "wb"))

    # ----------------------------
    # Vectorize column
    # ----------------------------
    def vectorise(self, col_name):
        if self.new_df is None:
            self.get_df()
        cv = CountVectorizer(max_features=5000, stop_words="english")
        vectors = cv.fit_transform(self.new_df[col_name]).toarray()
        return cosine_similarity(vectors)

    # ----------------------------
    # Build similarity file
    # ----------------------------
    def build_similarity_file(self, col_name):
        file_path = f"Files/similarity_tags_{col_name}.pkl"
        if not os.path.exists(file_path):
            sim = self.vectorise(col_name)
            pickle.dump(sim, open(file_path, "wb"))

    # ----------------------------
    # Boot function
    # ----------------------------
    def main_(self):
        self.get_df()
        for col in ["tags", "genres", "keywords", "tcast", "tprduction_comp"]:
            self.build_similarity_file(col)

    # ----------------------------
    # Getter
    # ----------------------------
    def getter(self):
        return self.new_df, self.movies, self.movies2