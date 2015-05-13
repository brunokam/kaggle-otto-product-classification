from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
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
params = [[8], [0.3]]
params_space = []
for i in xrange(len(params[0])):
    for j in xrange(len(params[1])):
        params_space.append([params[0][i], params[1][j]])

# Grid search
grid_errors = []
for params in params_space:

    # Cross validation
    skf = StratifiedKFold(y, 4)
    errors = []
    for train, test in skf:
        train_X = X[train]
        train_Y = y[train]
        test_X = X[test]
        test_Y = y[test]
        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_test = xgb.DMatrix(test_X, label=test_Y)

        # Setup parameters
        param = {'silent': 1, 'nthread': 2, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': 9,
                 'max_depth': params[0], 'eta': params[1]}
        num_round = 60

        # Train
        bst = xgb.train(param, xg_train, num_round, [(xg_train, 'train'), (xg_test, 'test')])
        # Predict
        predictions = bst.predict(xg_test).reshape(test_Y.shape[0], 9)
        # Get error
        errors.append(log_loss(test_Y, predictions))

    # Append new grid error
    grid_errors.append(np.mean(errors))

# Show results
for i in xrange(len(params_space)):
    print "Params: %s, logloss: %f" % (str(params_space[i]), grid_errors[i])
