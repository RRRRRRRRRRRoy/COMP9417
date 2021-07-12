#####################################################################


#####################################################################
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

#####################################################################
# This part of code is for Question 1 (b)
#####################################################################
c_grid = np.linspace(0.0001, 0.6, 100)
# Generate 100 number between 0.0001 and 0.6
c_grid = c_grid.tolist()
# print(c_grid)

Orginal_data = pd.read_csv("./hw02/Q1.csv")
# print(Orginal_data)
column_training_X = Orginal_data.iloc[:, 0: 45]
column_training_Y = Orginal_data.iloc[:, 45: 46]
Original_Training_X = np.array(column_training_X)
Original_Training_Y = np.array(column_training_Y)
# print(Original_Training_X)
# print(Original_Training_Y)
# print(Original_Training_Y.shape)
# print(Original_Training_X.shape)

train_X = Original_Training_X[:500]
train_Y = Original_Training_Y[:500]
test_X = Original_Training_X[500:]
test_Y = Original_Training_Y[500:]
# print(train_X.shape)
# print(train_Y.shape)
# print(test_X.shape)
# print(test_Y.shape)

log_loss_total_list = list()
log_loss_total_1000 = list()
# Greedy search is focused on train x and train y
for clf_in_grid in c_grid:
    classifier = LogisticRegression(C=clf_in_grid, solver='liblinear')
    log_losss_inside = list()
    for i in range(10):
        # Getting the matrix
        temp_X = np.copy(train_X)
        temp_Y = np.copy(train_Y)
        # Getting the testing data in the cross validation
        Valid_X = temp_X[i*50:(i+1)*50]
        Valid_Y = temp_Y[i*50:(i+1)*50]
        # Removing the data from the array
        Training_X = np.delete(
            temp_X, [del_i for del_i in range(i*50, (i+1)*50)], 0)
        Training_Y = np.delete(
            temp_Y, [del_i for del_i in range(i*50, (i+1)*50)], 0)
        classifier.fit(Training_X, np.ravel(Training_Y))
        y_predict = classifier.predict(Valid_X)
        log_loss_result = log_loss(Valid_Y, y_predict)
        log_losss_inside.append(log_loss_result)
        log_loss_total_1000.append(log_loss_result)
    log_loss_total_list.append(log_losss_inside)
# print(len(log_loss_total_list))
# print(log_loss_total_list)

log_loss_total_array = np.array(log_loss_total_list)
# print(log_loss_total_array.shape)
# print(log_loss_total_array)

log_loss_mean = np.mean(log_loss_total_array, axis=1)
# print(log_loss_mean)
least_log_mean_list = log_loss_mean.tolist()
least_log_loss = min(least_log_mean_list)
# print(least_log_loss)
C_index = least_log_mean_list.index(least_log_loss)
# print(C_index)
the_best_C = c_grid[C_index]
print(
    f"The best C is {the_best_C}, which log loss value is {least_log_mean_list[C_index]}")

# plt.boxplot(log_loss_total_list)
# plt.show()

# Here is the refit process
classifier_with_best_C = LogisticRegression(C=the_best_C, solver='liblinear')
classifier_with_best_C.fit(train_X, np.ravel(train_Y))
y_predict_with_best_C = classifier_with_best_C.predict(test_X)
training_acc = accuracy_score(train_Y, classifier_with_best_C.predict(train_X))
testing_acc = accuracy_score(test_Y, y_predict_with_best_C)
print(f"The Train accuracy is {training_acc}")
print(f"The Test accuracy is {testing_acc}")

#####################################################################
# This part of code is for Question 1 (c)
#####################################################################
