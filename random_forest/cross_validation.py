from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from utils import *

# Read data
X, y = get_train_data("../data/train.csv")

# Run cross validation
kf = KFold(y, n_folds=8)
y_pred = y * 0
fold = 1
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    clf = RandomForest(n_estimators=100, n_jobs=3)
    print "Fold %d: fitting" % fold
    clf.fit(X_train, y_train)
    print "Fold %d: predicting" % fold
    y_pred[test] = clf.predict(X_test)
    fold += 1

# Show report with accuracy for classes
print classification_report(y, y_pred, target_names=class_names)
