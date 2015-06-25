from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from utils import *
import sys

sys.path.append('D:\\libs\\xgboost\\wrapper')
import xgboost as xgb

# Read data
X, y = get_train_data("../data/train.csv")

# Features preprocessing
# pca = PCA(n_components=30)  # n_components='mle'
# X = pca.fit_transform(X)
# print "%s: %d features left" % (str(pca), len(pca.explained_variance_ratio_))

# Parameters space creation
params = [[8], [0.02]]
params_space = []
for i in xrange(len(params[0])):
    for j in xrange(len(params[1])):
        params_space.append([params[0][i], params[1][j]])

# Grid search
grid_errors = []
grid_best_iterations = []
for params in params_space:

    # Cross validation
    skf = StratifiedKFold(y, 10)
    errors = []
    best_iterations = []
    for train, test in skf:
        train_X = X[train]
        train_Y = y[train]
        test_X = X[test]
        test_Y = y[test]
        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_test = xgb.DMatrix(test_X, label=test_Y)

        # Setup parameters
        param = {'silent': 1, 'nthread': 2, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': 9,
                 'max_depth': params[0], 'eta': params[1]}  # , 'subsample': 0.5
        n_rounds = 4000  # Just a big number to trigger early stopping and best iteration

        # Train
        bst = xgb.train(param, xg_train, n_rounds, [(xg_train, 'train'), (xg_test, 'test')], early_stopping_rounds=20)
        # Predict
        predictions = bst.predict(xg_test).reshape(test_Y.shape[0], 9)
        # Get error and best iteration
        errors.append(log_loss(test_Y, predictions))
        best_iterations.append(bst.best_iteration)

    # Append new grid error
    grid_errors.append(np.mean(errors))
    grid_best_iterations.append(list(best_iterations))

# Show results
for i in xrange(len(params_space)):
    print "Params: %s, logloss: %f, best iterations: %s, mean: %f" % (
        str(params_space[i]), grid_errors[i], str(grid_best_iterations[i]), np.mean(grid_best_iterations[i]))
