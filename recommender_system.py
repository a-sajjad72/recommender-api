from time import time
print("importing modules")
st = start = time()

# import pandas as pd
from pandas import read_json, concat, Series, get_dummies, DataFrame
# import numpy as np
from numpy import savez_compressed
from thefuzz import process
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import os

timestamp = (
    lambda seconds: f"{int(seconds // 60)}m {round(seconds % 60, 2):.2f}s"
    if seconds >= 60
    else f"{round(seconds, 2):.2f}s"
    if seconds >= 1
    else f"{round(seconds * 1000):.0f}ms"
)

print(f"modules imported in {timestamp(time()-start)}\n")


def export_recommendation_engine(cos_sims, df, dir="recommendation_engine_data"):
    if not os.path.exists(dir):
        os.mkdir(dir)
    savez_compressed(
        os.path.join(dir, "recommender_engine"),
        cos_sims=cos_sims,
        books=df[["id", "title"]].to_numpy(),
    )


def get_data(file):
    return read_json("books-all(cleaned).json")[["id", "title", "language", "genres", "authors"]].head(2500)
    # return read_json(file)[["id", "title", "language", "genres", "authors"]]



# Data Cleaning: the genres are in the form of list of genres (dictionaries). each dictionary represents genre and some other details which irreleavant for model training
# from each dictionary we just need to get value of key `name`.
def clean_genres(df_genres):
    return df_genres.apply(
        lambda genres: [genre["name"].strip(" *") for genre in genres]
    )


# Data Cleaning: the authors are in the form of list of authors (dictionaries). each dictionary represents author and some other details which irreleavant for model training
# from each dictionary we just need to get values of keys `first_name` and `last_name`
def clean_authors(df_authors):
    return df_authors.apply(
        lambda authors: [
            " ".join(list(author.values())[1:3]).strip() for author in authors
        ]
    )


# get each author and no. of books in key and value respectively from the data.
def authors_counts(df_authors):
    return Counter(author for authors in df_authors for author in authors)


# get each genre and no of books associated with each genre in key and value respectively from the data.
def genres_counts(df_genres):
    return Counter(genre for genres in df_genres for genre in genres)


# transforming and one-hot encoding the genres column to a columnar format so that each column represents each genre name in dataframe
def encode_genres(df_genres, df_features):
    return concat(
        [
            df_features,
            df_genres.apply(
                lambda x: Series(
                    {g: int(g in x) for g in genres_counts(df_genres).keys()}
                )
            ),
        ],
        axis=1,
    )


# transforming and one-hot encoding the authors column to a columnar format so that each column represents each author name in dataframe
def encode_authors(df_authors, df_features):
    return concat(
        [
            df_features,
            df_authors.apply(
                lambda x: Series(
                    {g: int(g in x) for g in authors_counts(df_authors).keys()}
                )
            ),
        ],
        axis=1,
    )


# transforming and one-hot encoding the language column to a columnar format so that each language represents each column in dataframe
def encode_language(df_language, df_features):
    return concat([df_features, get_dummies(df_language)], axis=1)


# return the cosine similarities of the dataframe
def get_cosine_sims(df_features):
    return cosine_similarity(df_features, df_features)


def book_finder(title, df_title):
    return process.extractOne(title, df_title)[2]


def get_recommendations(df_main, cos_sims, title, n_recommendations=10):
    sim_scores = list(enumerate(cos_sims[book_finder(title, df_main.title)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : (n_recommendations + 1)]
    similar_books = [i[0] for i in sim_scores]
    return df_main[["id", "title"]].iloc[similar_books]


if __name__ == "__main__":
    # fields: id, title, language, author, genres
    # We will be focusing on utilizing above specified fields, importing them into our dataframe.
    # This selective import of fields ensures that we only work with the relevant data,
    # preventing any unnecassary data from being imported into the dataframe.
    print("importing data")
    start = time()
    books = get_data("books-all(cleaned).json")
    end = time()
    print(f"data imported in {timestamp(time()-start)}\n")

    # cleaning the genres column
    print("cleaning data")
    start = time()
    books["genres"] = clean_genres(books.genres)

    # cleaning the authors column
    books["authors"] = clean_authors(books.authors)
    print(f"data cleaned in {timestamp(time()-start)}\n")

    # initializing dataframe for features
    books_features = DataFrame()
    print("encoding data")
    s = start = time()
    # one-hot encoding languages
    books_features = encode_language(books.language, books_features)
    print(f"languages encoded in {timestamp(time()-start)}\n")

    # one-hot encoding genres
    start = time()
    books_features = encode_genres(books.genres, books_features)
    print(f"genres encoded in {timestamp(time()-start)}\n")

    # one-hot encoding authors
    start = time()
    books_features = encode_authors(books.authors, books_features)
    print(f"authors encoded in {timestamp(time()-start)}\n")

    print(f"data encoded in {timestamp(time()-s)}\n")
    # removing the language and authors column from our main dataframe because it is no more needed (memory constraints)
    # books.drop(["language", "authors"], axis=1, inplace=True)

    # calculating cosine similarities
    print("similarities calculation started")
    start = time()
    cosine_sims = get_cosine_sims(books_features)
    print(f"similarities calculated in {timestamp(time()-start)}\n")

    print("exporting data")
    start = time()
    export_recommendation_engine(cosine_sims, books)
    end = time()
    print(f"data exported in {timestamp(end-start)}\n")
    print(f"Process Done in {timestamp(end-st)}\n")

    # while True:
    #     name = input("enter the book name to get recommendations: ")
    #     rec = get_recommendations(books, cosine_sims, name)
    #     for x in rec:
    #         print(x)
