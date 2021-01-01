import os
import pandas as pd
import numpy as np
import scipy.sparse as sps
import sys
sys.path.append('../../')

import matplotlib.pyplot as plt

#data which contains users (row), items(col) and implicit interaction (data)
dataset = pd.read_csv('../data_train.csv')
dataset

users = dataset.row
items = dataset.col
data = dataset.data
URM_all = sps.coo_matrix((data, (users, items)))
URM_all = URM_all.tocsr() #fast row access -> fast access to users
URM_all.shape

ICM_df = pd.read_csv('../data_ICM_title_abstract.csv')
ICM_df

items = ICM_df.row
features = ICM_df.col
data = ICM_df.data
ICM_all = sps.coo_matrix((data, (items, features)))
ICM_all = ICM_all.tocsr() #fast row access -> fast access to users
ICM_all.shape

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Base.Evaluation.Evaluator import EvaluatorHoldout

URM_train_1, URM_validation_1 = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=1)

URM_train_2, URM_validation_2 = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=2)

URM_train_3, URM_validation_3 = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=3)

URM_train_4, URM_validation_4 = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=4)

URM_train_5, URM_validation_5 = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=5)

training_list = [URM_train_1,URM_train_2,URM_train_3,URM_train_4,URM_train_5]

evaluator_validation_1 = EvaluatorHoldout(URM_validation_1, cutoff_list=[10])
evaluator_validation_2 = EvaluatorHoldout(URM_validation_2, cutoff_list=[10])
evaluator_validation_3 = EvaluatorHoldout(URM_validation_3, cutoff_list=[10])
evaluator_validation_4 = EvaluatorHoldout(URM_validation_4, cutoff_list=[10])
evaluator_validation_5 = EvaluatorHoldout(URM_validation_5, cutoff_list=[10])

evaluator_list = [evaluator_validation_1, evaluator_validation_2, evaluator_validation_3, evaluator_validation_4, evaluator_validation_5]

from SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet

i = 1
for URM_train, evaluator in zip(training_list, evaluator_list):
    URM_train_ICM_all_SLIM = sps.vstack([URM_train * (1 - 0.421), ICM_all.T * 0.421])

    recommender = MultiThreadSLIM_ElasticNet(URM_train_ICM_all_SLIM)

    recommender.fit(alpha=0.000182, l1_ratio=4.45e-06, topK=996)
    recommender.W_sparse.data = np.power(recommender.W_sparse.data, 0.8546)

    # recommender.URM_train = URM_train.tocsr()

    result_dict, _ = evaluator.evaluateRecommender(recommender)
    print(result_dict)

    filename = 'SLIM_EN_fold_number' + str(i)
    recommender.save_model('SLIM_saved/', file_name=filename)
    i = i + 1