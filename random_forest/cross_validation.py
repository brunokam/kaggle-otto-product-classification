from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn import cross_validation
from utils import *

# Read data
X, y = get_train_data("../data/train.csv")

# Parameters to test
parameter_space = [[340]]

# Cross validation
parameter_scores = []
for parameter in parameter_space:
    clf = RandomForest(n_estimators=parameter[0], n_jobs=2)  # criterion='entropy'
    scores = cross_validation.cross_val_score(clf, X, y, cv=6, scoring='log_loss', verbose=3)
    parameter_scores.append(np.mean(scores * -1))

# Show results
print "Logloss: " + str(parameter_scores)
