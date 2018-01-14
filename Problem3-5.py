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

# same as previous and next file. So skip the file reading part
file_ = pd.read_csv("auto-mpg.data.txt", sep='\s+', header = None)
feature_map = ["mpg", "cylin", "disp", "horsep", "weight", "accele", "year", "origin", "name"]
file_.columns = feature_map
file_ = file_.replace({'?': np.nan}).dropna()

table = file_.sort_values("mpg", axis = 0, ascending = True, kind = 'heapsort')
table.reset_index(drop=True, inplace=True)

file_['horsep'] = pd.to_numeric(file_['horsep'].str.replace(' ', ''), errors='force')
table['horsep'] = pd.to_numeric(table['horsep'].str.replace(' ', ''), errors='force')


# Problem 3

# OLS function definition
def OLS(X, Y, degree):

    (row, col) = X.shape # get the col and row number of the X vector
    weights = np.zeros((col, degree + 1)) # initialize weights
    for i in range(0, col):
        matrix = helper(degree, i, X) # call helper function to return the MSE
        matrix_trans = np.transpose(matrix) # get the transpose of the matrix
        weights[i, :] = np.dot(np.linalg.pinv(np.dot(matrix_trans, matrix)), np.dot(matrix_trans, Y))
        # update the weights
    return weights



# Problem 4
deg = 4
f_num = 7

# MSE function definition
def MSE(X, Y, weights):
    predicted = np.dot(X, weights)
    return np.dot(np.transpose(predicted - Y), (predicted - Y))/(Y.shape)

def helper(degree, cc, X):
    # helper function to return the MSE
    (row, col) = X.shape
    buff = np.zeros((row, degree + 1))
    for i in range(0, degree + 1):
        buff[:, i] = X.iloc[:, cc].pow(i, axis=0)
    return buff

# split the dataset into 300 + 92(training, testing)
train, test = file_.iloc[:300, :], file_.iloc[300:, :]
train_y, train_x = train.iloc[:, 0], train.iloc[:, 1:-1]
test_y, test_x = test.iloc[:, 0], test.iloc[:, 1:-1]
# initialization of MSE_TR(train) and MSE_TE(test)
MSE_TR, MSE_TE = np.zeros((deg + 1, f_num)), np.zeros((deg + 1, f_num))
(row, col) = train_x.shape
#print("row", row, "col:", col)

# In first part of the question, we will need to calculate the MSE of all the values
for i in range(0, deg + 1):
    W = OLS(train_x, train_y, i)
    for j in range(0, col):
        mat_train = helper(i,j,train_x)
        mat_test = helper(i,j,test_x)
        MSE_TR[i, j] = MSE(mat_train, train_y, W[j]) # Training MSE
        MSE_TE[i, j] = MSE(mat_test, test_y, W[j]) # Testing MSE
#print("Training: ")
#print(MSE_TR)
#print("Testing: ")
#print(MSE_TE)

# Second part we need to plot graphs
for i in range(0, f_num + 1):

    W = OLS(train_x, train_y, 0)
    # print(w)
    x0 = test_x.iloc[:,i]
    y0 = test_y

    x = np.linspace(x0.min(), x0.max(), 92)

    r0 = W[i, 0]*x**0
    # Calculate OLS of different order and different x and y.
    # Using the general polynomial formula to do this part
    W = OLS(train_x, train_y, 1)
    r1 = W[i, 0]*x**0 + W[i, 1]*x**1
    # print(r1)
    W = OLS(train_x, train_y, 2)
    r2 = W[i, 0]*x**0 + W[i, 1]*x**1 + W[i, 2]*x**2
    # print(r2)
    W = OLS(train_x, train_y, 3)
    r3 = W[i, 0]*x**0 + W[i, 1]*x**1 + W[i, 2]*x**2 + W[i, 3]*x**3
    # print(r3)
    W = OLS(train_x, train_y, 4)
    r4 = W[i, 0]*x**0 + W[i, 1]*x**1 + W[i, 2]*x**2 + W[i, 3]*x**3 + W[i, 4]*x**4
    # print(r4)

    # construct the graph
    feature_graph = pd.DataFrame({"Testing set x": x0, "Testing set y": y0, "zero": r0, "first": r1, "second": r2, "third": r3, "fourth": r4, "x": x})
    # set up the graph axies
    axes = feature_graph.plot(kind = "scatter", x = "Testing set x", y = "Testing set y")
    feature_graph.plot(x = "Testing set x", y = "zero", color = "pink", label = "0", ax = axes)
    feature_graph.plot(x = "x", y = "first", color = "red", label = "1", ax = axes)
    feature_graph.plot(x = "x", y = "second", color = "green", label = "2", ax = axes)
    feature_graph.plot(x = "x", y = "third", color = "blue", label = "3", ax = axes)
    feature_graph.plot(x = "x", y = "fourth", color = "cyan", label = "4", ax = axes)
    # plot all the five graphs
    plt.show()

# Problem 5
deg = 2
MSE_TR = np.zeros((3)) # Reinitialization
MSE_TE = np.zeros((3))

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

for i in range(0, deg + 1):
    W = para_OLS(train_x, tran_y, i)
    mat_train = para_helper(train_x, 1, order)  # Poly matrix of both training and testing
    mat_test = para_helper(train_x, 1, order)
    MSE_TR[i] = MSE(mat_train, train_y, W)  # MSE Values
    MSE_TE[i] = MSE(mat_test, train_y, W)

#print(MSE_TR, MSE_TE)

# Please see problem6-7.py
