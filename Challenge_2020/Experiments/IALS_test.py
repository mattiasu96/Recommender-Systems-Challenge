import os

os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1

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

from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])


from MatrixFactorization.IALSRecommender_implicit import IALSRecommender_implicit

URM_train_ICM_all_IALS = sps.vstack([URM_all*(1-0.51668), ICM_all.T*0.51668])
recommender_IALS = IALSRecommender_implicit(URM_train_ICM_all_IALS)
recommender_IALS.fit(n_factors = 601, regularization = 0.8715, iterations=83, num_threads=4)