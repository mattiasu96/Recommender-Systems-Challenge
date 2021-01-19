# Recommender Systems 2020 Challenge

This repo contains the code and the data used in the **Recommender Systems 2020 Challenge** @ Politecnico di Milano. <br> Our tested models + challenge code can be found inside the [Challenge_2020](/Challenge_2020) folder. <br> All the remaining folders are library codebase forked by the [course repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi), which contains basic implementations of many recommenders + utility code.

We ended up with the following placement:

1. **Public leaderboard:** 3/66
2. **Private leaderboard:** 5/66

## Added functionalities

We added a couple of changes and useful extra code:

1) **GPU - Cython MF_IALS:** <br> The original [MF_IALS](/MatrixFactorization/algorithm/IALSRecommender.py) algorithm was quite slow, so we implemented a faster version which leaverages on GPU using the [implicit library](https://github.com/benfred/implicit). We adapted the code provided in the implicit library to match the same interface of our course repository, thus we were able to use the already implemented evaluator, data strctures ecc... with little extra effort. The GPU implementation allowed us to move from a 10 minutes per fit to about 30 seconds per fit, which is a huge improvement in performances.

2) **N score hybrid recommender**: <br> We extended the original [hybrid score recommender](/KNN/ItemKNNScoresHybridRecommender.py) (which merges only two recommenders) to an arbitrary number of number of recommender. The code can be found in [ItemKNNScoresHybridNRecommender.py](/KNN/ItemKNNScoresHybridNRecommender.py). <br> **NB:** there are other classes like [ItemKNNScoresHybrid5Recommender.py](/KNN/ItemKNNScoresHybrid5Recommender.py) which are "noise" from our various experiments. As the name suggest, this hybrid merges 5 recommenders. We higly suggest to use the generalized version [ItemKNNScoresHybridNRecommender.py](/KNN/ItemKNNScoresHybridNRecommender.py) which is simpler, cleaner and more flexible.

## Best model

Our best model merges three different algoritms:
1) [MF_IALS](/MatrixFactorization/algorithm/IALSRecommender.py) 
2) [RP3_Beta](/GraphBased/RP3betaRecommender.py) 
3) [SLIM_ElasticNet](/SLIM_ElasticNet/SLIMElasticNetRecommender.py) 

Each one of the above mentioned algorithms has been trained using the **Feature Merging** technique, which basically consists in merging the URM (user rating matrix) and the ICM (item content matrix) together and then training the model on this new matrix.

The best models can be found in [Best Models](/Challenge_2020/Best_models), in particular the code of the above mentioned model is [MF_IALS+rp3+Slim_elasticNet_featuremerge_0.09917](/Challenge_2020/Best_models/MF_IALS+rp3+Slim_elasticNet_featuremerge_0.09917_test.ipynb). The other models in this folder, even if they have a higher test score, were achieving worse performances on the private leaderboard. 

# FAQ
This section aims at helping future students with possible FAQ and problems we faced during the competition.

1. **I don't know where to start, how I choose my recommender/model?**
