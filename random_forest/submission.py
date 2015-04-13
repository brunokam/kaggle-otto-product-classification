from sklearn.ensemble import RandomForestClassifier as RandomForest
from utils import *

# Read data
X_train, y_train = get_train_data("../data/train.csv")
X_test = get_test_data("../data/test.csv")

# Fit model and make predictions
clf = RandomForest(n_estimators=340, n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_test)

# Save submission to file
save_submission(predictions)
