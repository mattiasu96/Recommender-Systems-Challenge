import pandas as pd
import numpy as np
import scipy.sparse as sps
import os
import sys
import matplotlib.pyplot as plt
import optuna

sys.path.append('../../')

from SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

dataset = pd.read_csv('../data_train.csv')
dataset

users = dataset.row
items = dataset.col
data = dataset.data
URM_all = sps.coo_matrix((data, (users, items)))
URM_all = URM_all.tocsr() #fast row access -> fast access to users
URM_all.shape

test_users = pd.read_csv('../data_target_users_test.csv')
test_users

ICM_df = pd.read_csv('../data_ICM_title_abstract.csv')
ICM_df

items = ICM_df.row
features = ICM_df.col
data = ICM_df.data
ICM_all = sps.coo_matrix((data, (items, features)))
ICM_all = ICM_all.tocsr() #fast row access -> fast access to users
ICM_all.shape

URM_train_1, URM_validation_1 = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.80)
evaluator_validation_1 = EvaluatorHoldout(URM_validation_1, cutoff_list=[10])

URM_train_2, URM_validation_2 = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.80)
evaluator_validation_2 = EvaluatorHoldout(URM_validation_2, cutoff_list=[10])

URM_train_3, URM_validation_3 = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.80)
evaluator_validation_3 = EvaluatorHoldout(URM_validation_3, cutoff_list=[10])

URM_train_vector = [URM_train_1, URM_train_2, URM_train_3]
Evaluator_vector = [evaluator_validation_1, evaluator_validation_2, evaluator_validation_3]


class Objective(object):
    def __init__(self, URM_train_vector, ICM_all, Evaluator_vector):
        # Hold this implementation specific arguments as the fields of the class.
        self.URM_train_vector = URM_train_vector
        self.ICM_all = ICM_all
        self.Evaluator_vector = Evaluator_vector
        self.MAP_vector = np.zeros(3)

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        topK = trial.suggest_int('topK', 50, 1000)
        weight = trial.suggest_uniform('weight', 0.1, 0.9)
        l1_ratio = trial.suggest_loguniform('l1_ratio', 1e-6, 1e-2)
        alpha = trial.suggest_uniform('alpha', 1e-5, 1e-2)
        self.MAP_vector = np.zeros(3)
        i = 0
        for URM_train, evaluator_validation in zip(self.URM_train_vector, self.Evaluator_vector):
            URM_train_ICM_all = sps.vstack([URM_train * (1 - weight), ICM_all.T * weight])
            recommender = MultiThreadSLIM_ElasticNet(URM_train_ICM_all)
            recommender.fit(topK=topK, l1_ratio=l1_ratio, alpha=alpha)
            result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
            self.MAP_vector[i] = result_dict[10]["MAP"]
            i = i + 1

        print('printing self map vector: ', self.MAP_vector)
        MAP = np.sum(self.MAP_vector) / 3
        print('printing MAP: ', MAP)

        return MAP


# Execute an optimization by using an `Objective` instance.
study = optuna.create_study(direction='maximize')
study.optimize(Objective(URM_train_vector, ICM_all, Evaluator_vector), n_trials=50)

print(study.best_params)
