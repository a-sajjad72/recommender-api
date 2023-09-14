import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils
from flask import Flask, request, jsonify, render_template, redirect

import traceback
import argparse
from pathlib import Path
import os

project_folder = Path(__file__).parent.resolve()

parser = argparse.ArgumentParser(
    prog="recommendation_api.py",
    description="An API of content-base recommendation for Audiobooks.",
)

# if not want to use the default file/path use command-line
# argument to give the path of the recommendation engine to import
parser.add_argument(
    "-p",
    "--path",
    default="recommendation_engine_data/recommender_engine.npz",
    type=Path,
    dest="path",
    help="path of .npz file. default to ./recommendation_engine_data/recommender_engine.npz",
)
args = parser.parse_args()


# loads the recommendation engine into memory
# the recommendation engine is basically a compressed
# exported numpy array of cosine similarities dataframe containing
# books id and titles.
def load_recommendation_engine(path=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not exists")
    with np.load(path, allow_pickle=True) as npz:
        return npz["cos_sims"], pd.DataFrame(npz["books"], columns=["id", "title"])


cos_sims, books = load_recommendation_engine(os.path.join(project_folder, args.path))


# takes the id of book (int) or title of book (string) as query
# performs approximate string matching and returns the
# similarity ratio of matches along all the books
def book_search(query, end, start=0):
    # checking the query is either id or title. if the query is id
    # gets the title of coresponding id from dataset.
    if "int" in str(type(query)):
        title_index = books.id.searchsorted(query)
        # validating is the id is valid or not
        if title_index < books.shape[0] and books.id[title_index] == query:
            query = books.title[title_index]
        else:
            raise ValueError(
                f"Invalid id.\nThe book id {query} does not exist in the database."
            )
    return process.extract(
        query,
        books.title,
        processor=utils.default_process,
        scorer=fuzz.QRatio,
        limit=end,
    )[start:end]


# takes the id of book (int) or title of book (string) as query
# returns the dictionary (object) of cosine similarities for the query
def recommender(query, n_recommendations, offset):
    # checking the query is either id or title. if the query is id
    # check the id is valid or not.
    if "int" in str(type(query)):
        title_index = books.id.searchsorted(query)
        if not (title_index < books.shape[0] and books.id[title_index] == query):
            raise ValueError(
                f"Invalid id.\nThe book id {query} does not exist in the database."
            )
    else:
        title_index = book_search(query, 1)[0][2]
    sim_scores = list(enumerate(cos_sims[title_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[offset:(n_recommendations)]
    similar_books = [i[0] for i in sim_scores]
    return books[["id", "title"]].iloc[similar_books].to_dict(orient="records")


app = Flask(__name__)


# redirecting the request to the docs path if user request for root path.
@app.route("/")
def root_path():
    return redirect("/docs")


# path for hosted docs of api
@app.route("/docs")
def docs():
    return render_template("docs.html")


# api endpoint for books searching
@app.route("/api/books")
def get_books():
    book_name = request.args.get("name", type=str)
    book_id = request.args.get("id", type=int)
    n_books = request.args.get("n_books", type=int, default=10)
    offset = request.args.get("offset", type=int, default=0)

    if offset >= books.shape[0] or offset < 0:
        return jsonify({"error": "Invalid offset. Audiobooks could not be found"}), 400

    if n_books < 1:
        return jsonify({"error": "Invalid value for n_books"}), 400

    if not (book_name or book_id):
        return jsonify({"error": "either name or id is required"}), 400

    try:
        if book_id:
            results = book_search(book_id, n_books + offset, offset)
        elif book_name:
            results = book_search(book_name, n_books + offset, offset)
        less_similar = []
        more_similar = []
        for x in results:
            if x[1] >= 60:
                more_similar.append([books.id[x[2]], x[0]])
            else:
                less_similar.append([books.id[x[2]], x[0]])
        return jsonify({"moreSimilar": more_similar, "lessSimilar": less_similar})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
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


# api endpiont for reccommendations
@app.route("/api/recommendations")
def get_recommendations():
    book_title = request.args.get("name", type=str)
    book_id = request.args.get("id", type=int)
    offset = request.args.get("offset", default=0, type=int)
    n_recs = request.args.get("n_recs", default=10, type=int)
    if offset >= books.shape[0] or offset < 0:
        return (
            jsonify({"error": "Invalid offset.\nUnable to generate recommendations."}),
            400,
        )

    if n_recs < 1:
        return jsonify({"error": "Invalid value for n_books"}), 400

    if not (book_title or book_id):
        return jsonify({"error": "either name or id is required"}), 400
    try:
        if book_id:
            recs = recommender(book_id, n_recs + offset, offset)
        else:
            recs = recommender(book_title, n_recs + offset, offset)
        return jsonify(recs), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="localhost", port=3000, debug=True)
