from sklearn import svm
from sklearn import preprocessing
from utils import *

# Read data
X_train, y_train = get_train_data("../data/train.csv")
X_test = get_test_data("../data/test.csv")

# Preprocessing
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit model and make predictions
clf = svm.SVC(C=10, cache_size=1000, probability=True, verbose=1)
clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_test)

# Save submission to file
save_submission(predictions)
