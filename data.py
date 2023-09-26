import numpy as np
import pandas as pd
import os
from pathlib import Path

project_folder = Path(__file__).parent.resolve()

def load_recommendation_engine(path=None):
    """
    Load the recommendation engine data into memory.

    :param path: Path to the recommendation engine data file.
    :return: Tuple containing cosine similarities and book dataframe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not exists")
    
    with np.load(path, allow_pickle=True) as npz:
        return npz["cos_sims"], pd.DataFrame(
            npz["books"],
            columns=["_id", "title", "authors", "genres", "language"],
        )

cos_sims, books = load_recommendation_engine(
    os.path.join(
        project_folder, "recommendation_engine_data", "recommender_engine.npz"
    )
)
