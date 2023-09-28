from data import books, cos_sims
from search import book_search


def recommender(query, n_recommendations, offset):
    """
    Get book recommendations based on a query (title or ID).

    :param query: The query (title or ID) to find recommendations for.
    :param n_recommendations: Number of recommendations to retrieve.
    :param offset: Starting index of recommendations.
    :return: List of recommended book IDs.
    """
    if "int" in str(type(query)):
        title_index = books["_id"].searchsorted(query)

        if not (title_index < books.shape[0] and books["_id"][title_index] == query):
            raise ValueError(f"Invalid id {query}")
    else:
        title_index = book_search(query, None, single_match=True)[2]

    sim_scores = list(enumerate(cos_sims[title_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[offset:(n_recommendations)]
    similar_books = [i[0] for i in sim_scores]

    # return books[["_id", "title"]].iloc[similar_books].to_dict(orient="records")
    return books["_id"].iloc[similar_books].tolist()
