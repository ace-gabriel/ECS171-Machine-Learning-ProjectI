"""
   1. mpg:           continuous
   2. cylinders:     multi-valued discrete
   3. displacement:  continuous
   4. horsepower:    continuous
   5. weight:        continuous
   6. acceleration:  continuous
   7. model year:    multi-valued discrete
   8. origin:        multi-valued discrete
   9. car name:      string (unique for each instance)

"""
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# need to read csv file again
file_ = pd.read_csv("auto-mpg.data.txt", sep='\s+', header = None)
# set up the data frame
feature_map = ["mpg", "cylin", "disp", "horsep", "weight", "accele", "year", "origin", "name"]
file_.columns = feature_map
# replace and delete garbage data
file_ = file_.replace({'?': np.nan}).dropna()
# using heapsort to sort the value
table = file_.sort_values("mpg", axis = 0, ascending = True, kind = 'heapsort')
# reset file axis
table.reset_index(drop=True, inplace = True)


# Problem 6

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

# Take first 300 samples as training and last 92 as testing
# get the training_x and training_y (i.e. training independent and dependent var)
train, test = file_.iloc[:300, :], file_.iloc[300:, :]
train_y, train_x = train.iloc[:, 0], train.iloc[:, 1:-1]
test_y, test_x = test.iloc[:, 0], test.iloc[:, 1:-1]

# Build identifier list for both training set and testing set
train_classify_arr, test_classify_arr = [], []
# put value into list
for item in train_y:
    if item <= 18.6:
        train_classify_arr.append(0)
    elif item > 18.6 and item <= 26.8:
        train_classify_arr.append(1)
    else:
        train_classify_arr.append(2)
# put value into array
for item in test_y:
    if item <= 18.6:
        test_classify_arr.append(0)
    elif item > 18.6 and item <= 26.8:
        test_classify_arr.append(1)
    else:
        test_classify_arr.append(2)

# turn lists into usable array
np.asarray(train_classify_arr)
np.asarray(test_classify_arr)

# call fit function and predicted the result
clf = LogisticRegression(C=1e5, max_iter = 12500)
clf.fit(train_x, train_classify_arr)
predicted = clf.predict(test_x)
# return test and train precision
train_accuracy = clf.score(train_x, train_classify_arr) # 85.333%
test_accuracy = clf.score(test_x, test_classify_arr) # 64.130%

def para_OLS(X, Y, degree):
    # This OLS function allows the parallel calculation
    mat = para_helper(X, 1, degree)
    mat_trans = np.transpose(mat)
    return np.dot(np.linalg.pinv(np.dot(mat_trans, mat)), np.dot(mat_trans, Y))


def para_helper(X, start, degree):
    # This helper function allows the parallel calculation of MSE values
    (row, col) = X.shape
    start = 1
    mat = np.ones((row, 1 + col * degree))
    for i in range(1, degree + 1):
        temp = X.pow(i, axis = 0)
        end = 1 + i * col
        poly_mat[:, start:end] = temp
        start = end
    return mat

# Problem 7

# logistic regression method
test_set = np.asarray([6, 350, 180, 3700, 9, 80, 1])

clf.predict([test_set]) # returns [0] -> low

# using second order method
tst = np.asarray([1, 6, 350, 180, 3700, 9, 80, 1, 6**2, 350**2, 180**2, 3700**2, 9**2, 80**2, 1])
weights = para_OLS(train_x, train_y, 2)
mpg = np.dot(weights, tst) # smaller than 18.6
