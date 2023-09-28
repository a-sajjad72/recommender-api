from flask import Blueprint, request, jsonify, render_template, redirect
import traceback

from recommendation import recommender
from search import book_search, genre_search, author_search
from data import books

api = Blueprint("api", __name__)


@api.route("/")
def root_path():
    """
    Redirect the root path to the documentation.
    """
    return redirect("/docs")


@api.route("/docs")
def docs():
    """
    Route for hosting API documentation.
    """
    return render_template("docs.html")


@api.route("/api/genres")
def get_books_by_genres():
    """
    API endpoint for searching books by genres.
    """
    genre_names = request.args.getlist("name[]")
    n_books = request.args.get("n_books", type=int, default=10)
    offset = request.args.get("offset", type=int, default=0)

    if not genre_names:
        return (
            jsonify(
                error={
                    "name": "QueryParameterError",
                    "message": "name is required",
                }
            ),
            400,
        )

    if n_books < 1:
        n_books = 10

    try:
        totalResults, results = genre_search(genre_names, n_books + offset, offset)
        return jsonify(totalResults=totalResults, results=results)
    except (IndexError, ValueError) as e:
        return (
            jsonify(
                error={
                    "name": type(e).__name__,
                    "message": str(e),
                }
            ),
            400,
        )
    except Exception as e:
        return (
            jsonify(
                error={
                    "name": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


@api.route("/api/authors")
def get_books_by_authors():
    """
    API endpoint for searching books by authors.
    """
    author_names = request.args.getlist("name[]")
    n_books = request.args.get("n_books", type=int, default=10)
    offset = request.args.get("offset", type=int, default=0)

    if not author_names:
        return (
            jsonify(
                error={
                    "name": "QueryParameterError",
                    "message": "name is required",
                }
            ),
            400,
        )

    if n_books < 1:
        n_books = 10

    try:
        totalResults, results = author_search(author_names, n_books + offset, offset)
        return jsonify(totalResults=totalResults, results=results)
    except (IndexError, ValueError) as e:
        return (
            jsonify(
                {
                    "error": {
                        "name": type(e).__name__,
                        "message": str(e),
                    }
                }
            ),
            400,
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "name": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }
                }
            ),
            500,
        )


@api.route("/api/books")
def get_books_by_name():
    """
    API endpoint for searching books by name.
    """
    book_name = request.args.get("name", type=str)
    book_id = request.args.get("_id", type=int)
    n_books = request.args.get("n_books", type=int, default=10)
    offset = request.args.get("offset", type=int, default=0)

    if offset >= books.shape[0] or offset < 0:
        return (
            jsonify(
                error={
                    "name": "QueryParameterError",
                    "message": "Invalid offset. Audiobooks could not be found",
                },
            ),
            400,
        )

    if n_books < 1:
        n_books = 10

    if not (book_name or book_id):
        return (
            jsonify(
                error={
                    "name": "QueryParameterError",
                    "message": "either name or id is required",
                }
            ),
            400,
        )

    try:
        if book_id:
            totalResults, results = book_search(book_id, n_books + offset, offset)
        elif book_name:
            totalResults, results = book_search(book_name, n_books + offset, offset)

        less_similar = []
        more_similar = []

        for x in results:
            if x[1] >= 60:
                # more_similar.append([books._id[x[2]], x[0]])
                more_similar.append(books._id[x[2]])
            else:
                # less_similar.append([books._id[x[2]], x[0]])
                less_similar.append(books._id[x[2]])

        return jsonify(
            totalResults=totalResults,
            moreSimilar=more_similar,
            lessSimilar=less_similar,
        )

    except (IndexError, ValueError) as e:
        return (
            jsonify(
                {
                    "error": {
                        "name": type(e).__name__,
                        "message": str(e),
                    }
                }
            ),
            400,
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "name": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }
                }
            ),
            500,
        )


@api.route("/api/recommendations")
def get_recommendations():
    """
    API endpoint for getting book recommendations.
    """
    book_title = request.args.get("name", type=str)
    book_id = request.args.get("id", type=int)
    offset = request.args.get("offset", default=0, type=int)
    n_recs = request.args.get("n_recs", default=10, type=int)

    if offset >= books.shape[0] or offset < 0:
        return (
            jsonify(
                {
                    "error": {
                        "name": "IndexError",
                        "message": "offset not valid",
                    }
                }
            ),
            400,
        )

    if n_recs < 1:
        n_recs = 10

    if not (book_title or book_id):
        return (
            jsonify(
                error={
                    "name": "QueryParameterError",
                    "message": "either name or id is required",
                }
            ),
            400,
        )

    try:
        if book_id:
            recs = recommender(book_id, n_recs + offset, offset)
        else:
            recs = recommender(book_title, n_recs + offset, offset)

        return jsonify(totalResults=books.shape[0], recommendations=recs), 200

    except (IndexError, ValueError) as e:
        return (
            jsonify(
                {
                    "error": {
                        "name": type(e).__name__,
                        "message": str(e),
                    }
                }
            ),
            400,
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "name": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }
                }
            ),
            500,
        )
