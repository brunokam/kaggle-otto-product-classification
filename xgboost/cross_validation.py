from sklearn import svm, cross_validation, preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from utils import *
import xgboost as xgb

# Read data
X, y = get_train_data("../data/train.csv")

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
    param = {'silent': 1, 'nthread': 2, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': 9}
    # param['max_depth'] = 4
    # param['eta'] = 0.1
    num_round = 100
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]

    # Train
    bst = xgb.train(param, xg_train, num_round, watchlist)
    # Predict
    y_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 9)
    # Get error
    errors.append(log_loss(test_Y, y_prob))

# Show results
print "Logloss: " + str(errors)
print "Mean: " + str(np.mean(errors))
