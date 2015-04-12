from sklearn.ensemble import RandomForestClassifier as RandomForest
from utils import *

# Read data
X_train, y_train = get_train_data("../data/train.csv")
X_test = get_test_data("../data/test.csv")

# Fit model and make predictions
clf = RandomForest(n_estimators=100, n_jobs=3)
print "Fitting"
clf.fit(X_train, y_train)
print "Predicting"
print X_test[0]
predictions = clf.predict(X_test)

# Create submission file
save_submission_from_indexes(predictions)
