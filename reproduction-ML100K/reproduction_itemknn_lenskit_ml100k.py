# Import packages

from lenskit.als import BiasedMFScorer
from lenskit.batch import recommend
from lenskit.data import ItemListCollection, UserIDKey, load_movielens
from lenskit.knn import ItemKNNScorer
from lenskit.metrics import NDCG, RBP, RecipRank, RunAnalysis
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import SampleFrac, crossfold_users

import numpy as np

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
    

# TODO: add random seed
ml100k = load_movielens("Dataset/ml-100k")

# TODO add information about the data --> print interactions, users, items, etc.
# TODO prune data

# TODO add random seed
final_test_method = SampleFrac(0.10) # holdout method for final test (here 10% of the data)

# Initialize lists for training and test data
train_data = ItemListCollection(UserIDKey)
final_test_data = ItemListCollection(UserIDKey)

# Split data into train and test sets
for split in crossfold_users(ml100k, 1, final_test_method):
    train_data.add_from(split.train)
    final_test_data.add_from(split.test)

# TODO add random seed to SampleFrac
validation_split_method = SampleFrac(0.1111) # holdout method for validation split (here 11.11% of the training data)

# Initialize lists for pure training set and validation set
pure_train_data = ItemListCollection(UserIDKey)
validation_data = ItemListCollection(UserIDKey)

# Split train data into pure train set and validation set
for split in crossfold_users(train_data, 1, validation_split_method):
    pure_train_data.add_from(split.train)
    validation_data.add_from(split.test)

# TODO show information of the data sets: print number of interactions and users

# Downsample the data into different percentages of the training data
# TODO add random seed
downsample_method = SampleFrac(1.0 - 0.5)

downsampled_train_data = ItemListCollection(UserIDKey)


for split in crossfold_users(pure_train_data, 1, downsample_method):
    downsampled_train_data.add_from(split.train)

# TODO show information of the downsampled data

k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 150, 180, 200, 220, 240, 260, 280]


for k in k_values:
    model_iknn = ItemKNNScorer(k=k)
    pipe_iknn = topn_pipeline(model_iknn)

    # train model with downsampled data (new clone for each split)
    fit_iknn = pipe_iknn.clone()
    fit_iknn.train(downsampled_train_data)

    # generate recommendations
    recs = recommend(fit_iknn, validation_data.keys(), 10)


