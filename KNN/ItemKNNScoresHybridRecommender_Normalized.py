#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/11/20

@author: GitMasters
"""

from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class ItemKNNScoresHybridRecommender_Normalized(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender_Normalized"


    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(ItemKNNScoresHybridRecommender_Normalized, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        
        
    def fit(self, alpha = 0.5):
        self.alpha = alpha      


    def _compute_item_score(self, user_id_array, items_to_compute):
        
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)

        print('shape of item_weights:', item_weights_1.shape)
        item_weights_1_max = item_weights_1.max()
        item_weights_2_max = item_weights_2.max()

        if not item_weights_1.any() == 0:
            item_weights_1 /= item_weights_1_max
        if not item_weights_2.any() == 0:
            item_weights_2 /= item_weights_2_max


        item_weights = item_weights_1*self.alpha + item_weights_2*(1-self.alpha)

        return item_weights