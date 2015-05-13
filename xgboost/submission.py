from utils import *
import sys

sys.path.append('D:\\libs\\xgboost\\wrapper')
import xgboost as xgb

# Read data
X_train, y_train = get_train_data("../data/train.csv")
X_test = get_test_data("../data/test.csv")

# Create xbg matrices
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test)

# Setup parameters
param = {'silent': 1, 'nthread': 2, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': 9,
         'max_depth': 8, 'eta': 0.3}
num_round = 90
watchlist = [(xg_train, 'train')]

# Train
bst = xgb.train(param, xg_train, num_round, watchlist)

# Predict
y_prob = bst.predict(xg_test).reshape(X_test.shape[0], 9)

# Save submission to file
save_submission(y_prob, 'xgb.csv')
