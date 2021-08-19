import numpy as np
import pandas as pd  # not really needed, only for preference
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("./Final_Exam/Data/Q4_train.csv")
# print(data)
X = np.array(data.iloc[:, 1: 6])
# get the column contain y
y = np.array(data.iloc[:, 6: 7])
# print(X)
# print(y)
###########################################################
###########################################################
# Question 4 question c
###########################################################
###########################################################


def Loss_function(pred_y, y_truth):
    # Notice in the previous using sum instead of mean
    Difference_pred_truth = pred_y - y_truth
    Difference_square = np.square(Difference_pred_truth)
    current_loss = np.sum(Difference_square)
    return current_loss


def total_loss(X, y, Z, models):
    loss_sum = 0
    # Looping the model in the list
    for index in range(len(models)):
        # Finding the index where z equals to index
        index_checker = np.where(Z == index)
        data_X = X[index_checker]
        data_y = y[index_checker]
        X_shape_ = data_X.shape[0]
        if X_shape_ != 0:
            pass
        else:
            continue
        current_model = models[index]
        current_model_coef_ = current_model.coef_
        current_coef_T = current_model_coef_.T
        predict_y = data_X.dot(current_coef_T)
        # From this part is to compute the loss value
        loss_sum = loss_sum + Loss_function(predict_y, data_y)
    return loss_sum


# This part of code is from the instruction
# # Example, if M=1, we would just fit a single linear model
mod = LinearRegression().fit(X, y)
# all points would belong to a singlepartition.
Z = np.zeros(shape=X.shape[0])
# Wrapped the model in the list
model_list = [mod]
print(total_loss(X, y, Z, model_list))  # outputs 298.328178158043
print()

###########################################################
###########################################################
# Question 4 question d
###########################################################
###########################################################

test_data = pd.read_csv("./Final_Exam/Data/Q4_test.csv")
print(test_data)
X_test = np.array(test_data.iloc[:, 1: 6])
# get the column contain y
y_test = np.array(test_data.iloc[:, 6: 7])
print(X_test.shape)
print(y_test)


def find_partitions(X, y, models):
    """
    given M models, assigns points in X to one of the M partitions
    :param X: design matrix, n x p (np.array shape (n,p))
    :param y: response vector, n x 1 (np.array shape (n,1) or (n,))
    :param models: a list of M sklearn LinearRegression models for each of the partitions
    returns: Z, a np.array of shape (n,) assigning each of the points in X to one of M partitions
    """
    M = len(models)
    loss_mat = np.zeros((M, X.shape[0]))
    for i in range(M):
        loss_mat[i] = (
            (np.dot(X, models[i].coef_.T) - y) ** 2)
    Z = np.argmin(loss_mat, axis=0)
    return Z


Z = find_partitions(X_test, y_test, model_list)
print(total_loss(X_test, y_test, Z, model_list))  # outputs 54.3679338727877
