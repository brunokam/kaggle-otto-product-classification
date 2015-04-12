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
    print "Train data read successfully"
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
    print "Test data read successfully"
    return X


# Based on vector with predicted class for each product create the submission file
def save_submission_from_indexes(predictions):
    print "Writing submission"
    with open('submission.csv', 'wb') as submission:
        writer = csv.writer(submission, delimiter=',')
        # Write description row
        writer.writerow(['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
        for i in xrange(len(predictions)):
            # Initialize prediction row
            row = np.zeros(10, dtype=np.int)
            # Add product id
            row[0] = i + 1
            # Add prediction
            row[predictions[i] + 1] = 1
            # Write new row
            writer.writerow(row)
    print "Submission written successfully"
