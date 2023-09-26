# search.py
from rapidfuzz import fuzz, process, utils
from data import books


def book_search(q_book: str, end, start=0):
    """
    Search for books based on a query (title or ID).

    :param q_book: The query (title or ID) to search for.
    :param end: Number of results to retrieve.
    :param start: Starting index of results.
    :return: List of search results.
    """
    if "int" in str(type(q_book)):
        title_index = books._id.searchsorted(q_book)

        if title_index < books.shape[0] and books._id[title_index] == q_book:
            q_book = books.title[title_index]
        else:
            raise ValueError(
                f"Invalid id.\nThe book id {q_book} does not exist in the database."
            )

    return process.extract(
        q_book,
        books.title,
        processor=utils.default_process,
        scorer=fuzz.token_ratio if len(q_book.strip().split()) > 1 else fuzz.WRatio,
        limit=end,
    )[start:end]


def genre_search(q_genre, end, start=0):
    """
    Search for books by genre.

    :param q_genre: List of genre names to search for.
    :param end: Number of results to retrieve.
    :param start: Starting index of results.
    :return: Dictionary with search results and total results count.
    """
    genres_list = [
        process.extractOne(
            genre,
            books.genres.explode().unique(),
            processor=utils.default_process,
            scorer=fuzz.token_ratio if len(genre.strip().split()) > 1 else fuzz.WRatio,
        )[0]
        for genre in q_genre
    ]

    # data = books[["_id", "genres"]][
    data = books["_id"][
        books.genres.apply(lambda genres: any(genre in genres_list for genre in genres))
    ]

    if start >= data.shape[0]:
        raise IndexError("offset not valid")

    return {"totalResults": data.shape[0], "results": data[start:end].tolist()}
    # return {
    #     "totalResults": data.shape[0],
    #     "results": data.iloc[start:end].to_dict(orient="records"),
    # }


def author_search(q_author, end, start=0):
    """
    Search for books by author.

    :param q_author: List of author names to search for.
    :param end: Number of results to retrieve.
    :param start: Starting index of results.
    :return: Dictionary with search results and total results count.
    """
    authors_list = [
        process.extractOne(
            author,
            books.authors.explode().unique(),
            processor=utils.default_process,
            scorer=fuzz.token_ratio,
        )[0]
        for author in q_author
    ]

    # data = books[["_id", "authors"]][
    data = books["_id"][
        books.authors.apply(
            lambda authors: any(author in authors_list for author in authors)
        )
    ]

    if start >= data.shape[0]:
        raise IndexError("offset not valid")

    return {"totalResults": data.shape[0], "results": data[start:end].tolist()}
    # return {
    #     "totalResults": data.shape[0],
    #     "results": data.iloc[start:end].to_dict(orient="records"),
    # }
