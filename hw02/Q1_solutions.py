#####################################################################
# This is the code for COMP9417 HW02 Question1
# Some of the annotation is the checkpoint of each part of code
# The encironment is Ubuntu 18.4
#####################################################################
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tqdm import tqdm
#####################################################################
# This part of code is for Question 1 (b)
# 
# Here is the source of question 1b
# using linspace to generate c_grid-----> stable than using for
# Source: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
# 
# To get the log loss using proba!!!!
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 
# The reason why using proba -----> Probability
# Source: https://www.kaggle.com/dansbecker/what-is-log-loss
#####################################################################
c_grid = np.linspace(0.0001, 0.6, 100)
# Generate 100 number between 0.0001 and 0.6
# c_grid = c_grid.tolist()
# print(c_grid)

# There is the data processing part
Orginal_data = pd.read_csv("./hw02/Q1.csv")
# print(Orginal_data)
# get the column contain x
column_training_X = Orginal_data.iloc[:, 0: 45]
# get the column contain y
column_training_Y = Orginal_data.iloc[:, 45: 46]
Original_X = np.array(column_training_X)
Original_Y = np.array(column_training_Y)
# print(Original_Training_Y.shape)
# print(Original_Training_X.shape)

# Cutting the training and testing data from the Original Data
train_X = Original_X[:500]
train_Y = Original_Y[:500]
test_X = Original_X[500:]
test_Y = Original_Y[500:]
# print(train_X.shape)
# print(train_Y.shape)
# print(test_X.shape)
# print(test_Y.shape)

# This is to store the log_loss result for getting the best C and plot
log_loss_total_list = list()
# Greedy search is focused on train x and train y
for clf_in_grid in c_grid:
    # Notice here solver='liblinear', penalty='l1'
    # random_state=0 is easy to check the result
    classifier = LogisticRegression(
        C=clf_in_grid, solver='liblinear', penalty='l1', random_state=0)
    # Getting log loss for each turn
    log_losss_inside = list()
    for i in range(10):
        # This part of code is to implement the cross validation
        # This part of code is similar to HW1
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

        # Training process
        classifier.fit(Training_X, np.ravel(Training_Y))
        y_predict = classifier.predict_proba(Valid_X)
        log_loss_result = log_loss(Valid_Y, y_predict)
        log_losss_inside.append(log_loss_result)
    log_loss_total_list.append(log_losss_inside)
# print(len(log_loss_total_list))

log_loss_total_array = np.array(log_loss_total_list)
# print(log_loss_total_array)

log_loss_mean = np.mean(log_loss_total_array, axis=1)
# print(log_loss_mean)
least_log_mean_list = log_loss_mean.tolist()
least_log_loss = min(least_log_mean_list)
# print(least_log_loss)
C_index = least_log_mean_list.index(least_log_loss)
# C_index = np.where(log_loss_mean == least_log_loss)
# print(C_index)
the_best_C = c_grid[C_index]
print("Here is the result of question 1 (b)")
print(f"The best C is {the_best_C}")
print(f"The log loss value is {least_log_loss}")

fig = plt.figure(figsize=(20, 10))
fig.autofmt_xdate()
plt.boxplot(log_loss_total_list, labels=np.around(c_grid, decimals=4))
plt.xticks(rotation=90)
plt.show()


# Here is the refit process  total 500 training and testing
# Using the best C value to retrain the model
classifier_best_C = LogisticRegression(
    C=the_best_C, solver='liblinear', penalty='l1', random_state=0)
classifier_best_C.fit(train_X, np.ravel(train_Y))
y_predict_with_best_C = classifier_best_C.predict(test_X)
training_acc = accuracy_score(train_Y, classifier_best_C.predict(train_X))
testing_acc = accuracy_score(test_Y, y_predict_with_best_C)
# Here is the result of Train acc and Test acc
print(f"The Train accuracy is {training_acc}")
print(f"The Test accuracy is {testing_acc}")
print()
#####################################################################
# This part of code is for Question 1 (c)
# GridSearch CV is difference with the manual one
# It is based on the label to separate into different
# CV can separate based on y label, and the result is balance
# 
# here is the Source mentioned in Question 1 (C)
# The parameter of GridSearchCV
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# How to using Kfold in the parameter
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# How to use acc score
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
#####################################################################
param_grid = {"C": c_grid}
# This Grid Search CV method is from the Instruction PDF file
# Setting the random state to 0 is easy to check
grid_lr_1 = GridSearchCV(estimator=LogisticRegression(
    penalty='l1', solver='liblinear'), cv=10, param_grid=param_grid)
grid_lr_1.fit(train_X, np.ravel(train_Y))


getting_log_loss = grid_lr_1.predict_proba(test_X)
log_loss_result = log_loss(test_Y, getting_log_loss)


print("Here is the result of question 1 (c): No parameters changed")
# The best params are provided by searchCV
C_result_1 = grid_lr_1.best_params_
C_value_1 = C_result_1["C"]
print(f"The best C is {C_value_1}")
print(f"The Log loss of GridSearchCV is {log_loss_result}")


# This part of code is same as the previous
# Here changing the parameter to make the result same 
grid_lr = GridSearchCV(estimator=LogisticRegression(
    penalty='l1', solver='liblinear', random_state=0), 
    scoring='neg_log_loss', cv=KFold(10), param_grid=param_grid)
grid_lr.fit(train_X, np.ravel(train_Y))

predict_y_GCV = grid_lr.predict(test_X)
predict_y_GCV_train = grid_lr.predict(train_X)


# Here we can using the accuracy score to get the result of C
train_accuracy_GCV = accuracy_score(train_Y, predict_y_GCV_train)
test_accuracy_GCV = accuracy_score(test_Y, predict_y_GCV)
print()
print("Here is the result of (c): Chaning the Kfold and Scoring!")
# The best params are provided by searchCV
C_result = grid_lr.best_params_
C_value = C_result["C"]
print(f"The best C is {C_value}")
print(f"The Log loss of GridSearchCV is {grid_lr.best_score_}")
print(f"The Train accuracy of GridSearchCV is {train_accuracy_GCV}")
print(f"The Test accuracy of GridSearchCV is {test_accuracy_GCV}")
print()
#####################################################################
# This part of code is for Question 1 (d)
# In the following question C=1, using all training set
# boostrap implementation
# Source: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
# How to use percentile to get the data
# Source: https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
#####################################################################
print("Here is the plot of (d)")
coefficient_list = list()
np.random.seed(12)
for item in tqdm(range(10000)):
    # generating train-i
    # i random list --> range(0-499) len(500) !!! important !!!
    random_list = np.random.randint(0, 500, 500)
    boostrap_train_X = np.zeros_like(train_X)
    boostrap_train_Y = np.zeros_like(train_Y)
    for index in range(500):
        boostrap_temp_X = train_X[random_list[index]]
        boostrap_temp_Y = train_Y[random_list[index]]
        boostrap_train_X[index] = boostrap_temp_X
        boostrap_train_Y[index] = boostrap_temp_Y

    # Notice: Here is the same para in Question C
    classifier_d = LogisticRegression(
        C=1.0, solver='liblinear', penalty='l1')

    # Getting the parameters
    classifier_d.fit(boostrap_train_X, boostrap_train_Y.ravel())
    coefficient_result = classifier_d.coef_
    coefficient_list.append(coefficient_result)

# Purify the structure of the list
purify_coeffcient_list = list()
for item in coefficient_list:
    for i in item:
        purify_coeffcient_list.append(i)

# Getting the previous 9000 lines
coefficient_list_9000 = np.array(purify_coeffcient_list[:9000])
# print(coefficient_list_9000.shape)
# print(type(coefficient_list_9000))

# Getting the 50%
# fifty_column = np.percentile(purify_coeffcient_list, 50, axis=0)
# The question Said get mean for each bar
# Therefore using 9000 list to get the mean
mean_column = np.mean(purify_coeffcient_list, axis=0)
# # Getting the 5%
fifth_column = np.percentile(coefficient_list_9000, 5, axis=0)
# # Getting the 95%
ninety_fifth_column = np.percentile(coefficient_list_9000, 95, axis=0)

# print(fifth_column)
# print(ninety_fifth_column)

# Generating the data list to draw the plot
bar_data = list()
for index in range(len(fifth_column)):
    bar_data.append([fifth_column[index], ninety_fifth_column[index]])
# This is for debug
# print(bar_data)


# This is for debug
# print(np.max(ninety_fifth_column))
# print(np.min(fifth_column))

# Checking the color of each bar
# using the absolute length to draw the bar plot
# Checking the span of each bar and print the color
for index in range(len(bar_data)):
    ndarray_data = np.array(bar_data[index])
    # This is the red bar
    # Check the bottom and the height of each bar
    if fifth_column[index] <= 0 and ninety_fifth_column[index] >= 0:
        plot_draw = plt.bar(
            index, bottom=fifth_column[index], height=ninety_fifth_column[index]+abs(fifth_column[index]), color='red')
    else:
        # This is the blue bar
        if fifth_column[index] < 0 and ninety_fifth_column[index] < 0:
            plot_draw = plt.bar(
                index, bottom=ninety_fifth_column[index], height=fifth_column[index]-ninety_fifth_column[index], color='blue')
        if fifth_column[index] > 0 and ninety_fifth_column[index] > 0:
            plot_draw = plt.bar(
                index, bottom=fifth_column[index], height=ninety_fifth_column[index]-fifth_column[index], color='blue')
label_XXX = [i for i in range(len(mean_column))]

# Draw the mean dot on the graph
plt.plot(label_XXX,
         mean_column, "o", color="lime", label="Mean Point", markersize="3")
plt.legend()

plt.show()
