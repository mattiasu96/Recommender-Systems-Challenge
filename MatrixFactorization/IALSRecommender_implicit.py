"""
Created on 23/03/2019

@author: Maurizio Ferrari Dacrema
"""



from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.Recommender_utils import check_matrix
import implicit
from tqdm import tqdm
import numpy as np


class IALSRecommender_implicit(BaseMatrixFactorizationRecommender):
    """
    ALS implemented with implicit following guideline of
    https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
    IDEA:
    Recomputing x_{u} and y_i can be done with Stochastic Gradient Descent, but this is a non-convex optimization problem.
    We can convert it into a set of quadratic problems, by keeping either x_u or y_i fixed while optimizing the other.
    In that case, we can iteratively solve x and y by alternating between them until the algorithm converges.
    This is Alternating Least Squares.
    """

    def __init__(self, URM_train, verbose = True):
        super(IALSRecommender_implicit, self).__init__(URM_train, verbose = verbose)

    def fit(self,  n_factors=300, regularization=0.15, iterations=30, num_threads=2):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations

        sparse_item_user = self.URM_train.T

        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=self.n_factors, regularization=self.regularization, iterations=self.iterations,  num_threads=num_threads)


        alpha_val = 24
        # Calculate the confidence by multiplying it by our alpha value.

        data_conf = (sparse_item_user * alpha_val).astype('double')

        # Fit the model
        model.fit(data_conf)

        # Get the user and item vectors from our trained model
        self.USER_factors = model.user_factors
        self.ITEM_factors = model.item_factors

