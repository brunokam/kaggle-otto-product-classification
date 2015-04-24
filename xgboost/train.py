from sklearn import svm, cross_validation
from sklearn import preprocessing
from utils import *
from ctypes import *
import xgboost as xgb

windll.LoadLibrary("D:/libs/xgboost-master/windows/x64/Release/xgboost_wrapper.dll")

# Read data
X, y = get_train_data("../data/train.csv")

# label need to be 0 to num_class -1
data = np.loadtxt('../data/train.csv', delimiter=',')
sz = data.shape
print sz
exit(1)


# Preprocessing
X = preprocessing.StandardScaler().fit_transform(X)

# Parameters to test
parameter_space = [[10], [14], [18], [22]]

# Cross validation
parameter_scores = []
for parameter in parameter_space:
    clf = svm.SVC(C=parameter[0], cache_size=1000, probability=True)
    scores = cross_validation.cross_val_score(clf, X, y, cv=2, scoring='log_loss', verbose=3)
    parameter_scores.append(np.mean(scores * -1))

# Show results
print "Logloss: " + str(parameter_scores)





#
#
# train = data[:int(sz[0] * 0.7), :]
# test = data[int(sz[0] * 0.7):, :]
#
# train_X = train[:,0:33]
# train_Y = train[:, 34]
#
#
# test_X = test[:,0:33]
# test_Y = test[:, 34]
#
# xg_train = xgb.DMatrix( train_X, label=train_Y)
# xg_test = xgb.DMatrix(test_X, label=test_Y)
# # setup parameters for xgboost
# param = {}
# # use softmax multi-class classification
# param['objective'] = 'multi:softmax'
# # scale weight of positive examples
# param['eta'] = 0.1
# param['max_depth'] = 6
# param['silent'] = 1
# param['nthread'] = 4
# param['num_class'] = 6
#
# watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
# num_round = 5
# bst = xgb.train(param, xg_train, num_round, watchlist );
# # get prediction
# pred = bst.predict( xg_test );
#
# print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
#
# # do the same thing again, but output probabilities
# param['objective'] = 'multi:softprob'
# bst = xgb.train(param, xg_train, num_round, watchlist );
# # Note: this convention has been changed since xgboost-unity
# # get prediction, this is in 1D array, need reshape to (ndata, nclass)
# yprob = bst.predict( xg_test ).reshape( test_Y.shape[0], 6 )
# ylabel = np.argmax(yprob, axis=1)
#
# print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
