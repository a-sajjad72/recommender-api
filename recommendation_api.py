import numpy as np
import pandas as pd

from rapidfuzz import fuzz, process, utils
from flask import Flask, request, jsonify

import traceback


def load_recommendation_engine(path=None):
    with np.load(
        path, allow_pickle=True
    ) as npz:
        return npz["cos_sims"], pd.DataFrame(npz["books"], columns=["id", "title"])


cos_sims, books = load_recommendation_engine("recommendation_engine_data/recommender_engine.npz")


def book_search(query, end, start=0):
    print("query type", str(type(query)))
    if "int" in str(type(query)):
        query = books.title[books.id.searchsorted(query)]

    return process.extract(
        query,
        books.title,
        processor=utils.default_process,
        scorer=fuzz.QRatio,
        limit=end,
    )[start:]


def recommender(query, n_recommendations, offset):
    if "int" in str(type(query)):
        title_idx = books.id.searchsorted(query)
    else:
        title_idx = book_search(query, 1)[0][2]

    sim_scores = list(enumerate(cos_sims[title_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[offset : (n_recommendations + 1)]
    similar_books = [i[0] for i in sim_scores]
    return books[["id", "title"]].iloc[similar_books].to_dict(orient="records")


app = Flask(__name__)


@app.route("/books")
def get_books():
    book_name = request.args.get("name", type=str)
    book_id = request.args.get("id", type=int)
    n_books = request.args.get("n_books", type=int, default=10)
    offset = request.args.get("offset", type=int, default=0)
    if not (book_name or book_id):
        return (
            jsonify({"error": "either name or id is required"}),
            400,
        )
    try:
        if book_id:
            results = book_search(book_id, n_books, offset)
        elif book_name:
            results = book_search(book_name, n_books, offset)

        less_similar = []
        more_similar = []
        for x in results:
            if x[1] >= 60:
                more_similar.append([x[0], books.id[x[2]]])
            else:
                less_similar.append([x[0], books.id[x[2]]])
        return jsonify({"moreSimilar": more_similar, "lessSimilar": less_similar})
    except Exception as e:
        return (
            jsonify(
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


@app.route("/recommendations")
def get_recommendations():
    book_title = request.args.get("name", type=str)
    book_id = request.args.get("id", type=int)
    offset = request.args.get("offset", default=1, type=int)
    n_recs = request.args.get("n_recs", default=10, type=int)
    if not (book_title or book_id):
        return jsonify({"error": "either name or id is required"}), 400

    if n_recs < 1:
        n_recs = 10

    try:
        if book_id:
            recs = recommender(book_id, n_recs, offset)
        else:
            recs = recommender(book_title, n_recs, offset)
        return jsonify(recommendations=recs), 200

    except Exception as e:
        return (
            jsonify(
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


app.run(host="localhost", port=3000, debug=True)
