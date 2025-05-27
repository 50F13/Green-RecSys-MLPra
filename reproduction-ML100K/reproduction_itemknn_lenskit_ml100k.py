# Import packages

from lenskit.als import BiasedMFScorer
from lenskit.batch import recommend
from lenskit.data import ItemListCollection, UserIDKey, load_movielens, Dataset, DatasetBuilder
from lenskit.knn import ItemKNNScorer
from lenskit.metrics import NDCG, RBP, RecipRank, RunAnalysis, ListLength
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import SampleFrac, crossfold_users
from lenskit import random

import numpy as np


# nDCG calculation from original code
class nDCG_LK:
    def __init__(self, n, top_items, test_items):
        self.n = n
        self.top_items = top_items
        self.test_items = test_items

    def _ideal_dcg(self):
        iranks = np.zeros(self.n, dtype=np.float64)
        iranks[:] = np.arange(1, self.n + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=0)
        if len(self.test_items) < self.n:
            idcg[len(self.test_items):] = idcg[len(self.test_items) - 1]
        return idcg[self.n - 1]

    def calculate_dcg(self):
        dcg = 0
        for i, item in enumerate(self.top_items):
            if item in self.test_items:
                relevance = 1
            else:
                relevance = 0
            rank = i + 1
            contribution = relevance / np.log2(rank + 1)
            dcg += contribution
        return dcg

    def calculate(self):
        dcg = self.calculate_dcg()
        ideal_dcg = self._ideal_dcg()
        if ideal_dcg == 0:
            return 0
        ndcg = dcg / ideal_dcg
        return ndcg
    
# random seed for the code, change seed_int to change the random seed in the entire code
seed_int = 42
random.set_global_rng(seed_int)

# load dataset from specified location in your files
ml100k = load_movielens("Dataset/ml-100k")

def main():

    # TODO add information about the data --> print interactions, users, items, etc.
    # TODO prune data

    # holdout method for final test (here 10% of the data)
    final_test_method = SampleFrac(0.10, rng = seed_int)

    # Initialize lists for training and test data
    final_test_data = ItemListCollection(UserIDKey)

    # Split data into train and test sets
    for split in crossfold_users(ml100k, 1, final_test_method):
        train_data_builder = DatasetBuilder(split.train)
        final_test_data.add_from(split.test)

    # creates train dataset
    train_data = train_data_builder.build()


    # holdout method for validation split (here 11.11% of the training data)
    validation_split_method = SampleFrac(0.1111, rng = seed_int)

    # Initialize lists for pure training set and validation set
    validation_data = ItemListCollection(UserIDKey)

    # Split train data into pure train set and validation set
    for split in crossfold_users(train_data, 1, validation_split_method):
        pure_train_data_builder = DatasetBuilder(split.train)
        validation_data.add_from(split.test)

    # creates the pure train data set
    pure_train_data = pure_train_data_builder.build()

    # TODO show information of the data sets: print number of interactions and users

    # Downsample the data into different percentages of the training data
    downsample_method = SampleFrac(1.0 - 0.5, rng = seed_int)

    # downsample the data
    for split in crossfold_users(pure_train_data, 1, downsample_method):
        downsampled_train_data_builder = DatasetBuilder(split.train)

    # creates the downsampled dataset
    downsampled_train_data = downsampled_train_data_builder.build()

    # TODO show information of the downsampled data

    # function that trains the pipeline and then evaluates the results
    def evaluate_with_ndcg(pipe, train_data, valid_data):
        # train pipeline
        fit_pipe = pipe.clone()
        fit_pipe.train(train_data)
        users = valid_data.keys()
        # generate recommendations for the validation data
        recs = recommend(fit_pipe, users, 10)

        ran = RunAnalysis()
        ran.add_metric(NDCG(10)) # calculates nDCG@10
        ndcg_result = ran.measure(recs, valid_data)
        list_metrics = ndcg_result.list_metrics()
        # print(f"List metrics: {list_metrics}")
        list_summary = ndcg_result.list_summary()

        # accesses the mean value of the ndcg measure for the recommendations
        mean_ndcg = list_summary.iloc[0, 0]


        # total_ndcg = 0
        # mean_ndcg = 0

        # # calculate ndcg for the recommendations
        # for user in users:
        #     user_recs = recs.lookup(user)
        #     user_truth = valid_data.lookup(user)
        #     ndcg_score = nDCG_LK(10, user_recs, user_truth).calculate()
        #     total_ndcg += ndcg_score
        
        # mean_ndcg = total_ndcg / valid_data.__len__()


        return mean_ndcg
    

    k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 150, 180, 200, 220, 240, 260, 280]
    results = []
    best_k = None
    best_mean_ndcg = -float('inf') # initialized to negative infinity (first ndcg value will be better)

    for k_kandidate in k_values:
        # create model and pipeline for item knn
        model_iknn = ItemKNNScorer(k=k_kandidate)
        pipe_iknn = topn_pipeline(model_iknn)
        
        # run the pipeline with training and validation data
        mean_ndcg = evaluate_with_ndcg(pipe_iknn, downsampled_train_data, validation_data)

        # document the results
        results.append({'K': k_kandidate, 'Mean nDCG': mean_ndcg})

        # evaluate the best k
        if mean_ndcg > best_mean_ndcg:
            best_mean_ndcg = mean_ndcg
            best_k = k_kandidate

    # print the results
    print("Results:")
    for result in results:
        print(f"K = {result['K']}: Mean nDCG = {result['Mean nDCG']:.4f}")

    print(f"\nBest K: {best_k} (Mean nDCG = {best_mean_ndcg:.4f})")

    # build the final model and pipeline for the test data
    final_model = ItemKNNScorer(k=best_k)
    final_pipe = topn_pipeline(final_model)

    final_mean_ndcg = evaluate_with_ndcg(final_pipe, downsampled_train_data, final_test_data)

    # print the results of the recommendation for the final test data
    print(f"nDCG mean for test set: {final_mean_ndcg:.4f}")


if __name__ == "__main__":
    main()