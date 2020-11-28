#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/11/20

@author: Arcangelo Pisa
"""

from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender


class ItemKNNSimilarityHybridMultipleRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNSimilarityHybridMultipleRecommender
    Hybrid of two similarities S = S1*alpha + S2*beta * S3*(1-alpha-beta)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridMultipleRecommender"


    def __init__(self, URM_train, Similarity_1, Similarity_2, Similarity_3, verbose = True):
        super(ItemKNNSimilarityHybridMultipleRecommender, self).__init__(URM_train, verbose = verbose)

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("ItemKNNSimilarityHybridMultipleRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        if Similarity_1.shape != Similarity_3.shape:
            raise ValueError("ItemKNNSimilarityHybridMultipleRecommender: similarities have different size, S1 is {}, S3 is {}".format(
                Similarity_1.shape, Similarity_3.shape
            ))

        if Similarity_2.shape != Similarity_3.shape:
            raise ValueError("ItemKNNSimilarityHybridMultipleRecommender: similarities have different size, S3 is {}, S2 is {}".format(
                Similarity_3.shape, Similarity_2.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')
        self.Similarity_3 = check_matrix(Similarity_3.copy(), 'csr')


    def fit(self, topK=100, alpha = 0.5, beta = 0.5):

        self.topK = topK
        self.alpha = alpha
        self.beta = beta

        W_sparse = self.Similarity_1*self.alpha + self.Similarity_2*self.beta + self.Similarity_3*(1-self.alpha-self.beta)

        self.W_sparse = similarityMatrixTopK(W_sparse, k=self.topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')