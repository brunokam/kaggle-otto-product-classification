from sklearn import svm, cross_validation
from sklearn import preprocessing
from utils import *

# Read data
X, y = get_train_data("../data/train.csv")

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
