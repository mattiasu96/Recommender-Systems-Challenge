#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/11/20

@author: GitMasters
"""

from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class ItemKNNScoresHybrid4Recommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*beta + R3*(1-alpha-beta)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybrid5Recommender"


    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3, Recommender_4):
        super(ItemKNNScoresHybrid4Recommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3
        self.Recommender_4 = Recommender_4

        
    def fit(self, first = 0.5, second = 0.5, third = 0.5):
        self.first = first    
        self.second = second
        self.third = third


    def _compute_item_score(self, user_id_array, items_to_compute):
        
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.Recommender_3._compute_item_score(user_id_array)
        item_weights_4 = self.Recommender_4._compute_item_score(user_id_array)
        print('shape of item_weights:', item_weights_1.shape)

        item_weights = item_weights_1*self.first + item_weights_2*self.second + item_weights_3*self.third + item_weights_4*(1 - (self.first + self.second + self.third))

        return item_weights