import numpy as np
import csv

class_names = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"]


# Read data from train.csv file into X matrix containing features and y vector containing labels
def get_train_data(path):
    print "Reading train data"
    data = np.array(list(csv.reader(open(path, "rb"), delimiter=',')))
    # Skip first row with data description
    data = data[1:]
    # Create X matrix without first column containing id and last column containing class label
    X = data[:, 1:-1]
    X = X.astype('int')
    # Create y vector from last column and convert to int
    y = data[:, -1]
    y = np.array([class_names.index(elem) for elem in y])
    return X, y


# Read data from test.csv file into X matrix containing features
def get_test_data(path):
    print "Reading test data"
    data = np.array(list(csv.reader(open(path, "rb"), delimiter=',')))
    # Skip first row with data description
    data = data[1:]
    # Create X matrix without first column containing id
    X = data[:, 1:]
    X = X.astype('int')
    return X


# Based on scores matrix write submission to file
def save_submission(scores):
    print "Writing submission"
    with open('submission.csv', 'wb') as submission:
        # Write description row
        submission.write('id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n')
        for i in xrange(len(scores)):
            # Write new row
            submission.write(str(i + 1))
            for elem in scores[i]:
                submission.write(',' + str(elem))
            submission.write('\n')
