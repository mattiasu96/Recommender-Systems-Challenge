import numpy as np
import implicit
import pandas as pd
from tqdm import tqdm
from Base.BaseRecommender import BaseRecommender

import numpy as np
from Base.Recommender_utils import check_matrix, similarityMatrixTopK

import scipy.sparse as sps
from sklearn.preprocessing import normalize
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
import time, sys

class AlternatingLeastSquare(BaseItemSimilarityMatrixRecommender):
    """
    ALS implemented with implicit following guideline of
    https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

    IDEA:
    Recomputing x_{u} and y_i can be done with Stochastic Gradient Descent, but this is a non-convex optimization problem.
    We can convert it into a set of quadratic problems, by keeping either x_u or y_i fixed while optimizing the other.
    In that case, we can iteratively solve x and y by alternating between them until the algorithm converges.
    This is Alternating Least Squares.

    """

    RECOMMENDER_NAME = "AlternatingLeastSquare"

    def __init__(self, URM_train, n_factors=300, regularization=0.15, iterations=30, verbose = True):
        super(AlternatingLeastSquare, self).__init__(URM_train, verbose = verbose)
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.URM_train = URM_train

    def fit(self):
        self.URM = self.URM_train.copy()

        sparse_item_user = self.URM.T

        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=self.n_factors, regularization=self.regularization, iterations=self.iterations)


        alpha_val = 24
        # Calculate the confidence by multiplying it by our alpha value.

        data_conf = (sparse_item_user * alpha_val).astype('double')

        # Fit the model
        model.fit(data_conf)

        # Get the user and item vectors from our trained model
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors


    def get_expected_ratings(self, user_id):
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)

        return np.squeeze(scores)

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

# recommender = AlternatingLeastSquare()
# Runner.run(True, recommender, None)
#
