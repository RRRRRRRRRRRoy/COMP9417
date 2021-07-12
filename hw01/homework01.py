###########################################################################################
# The difference between lasso and Ridge (import for writing hw1 report)
# Source: https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/
###########################################################################################
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import numpy as np
import seaborn as sbn
import pandas as pd
import matplotlib.pyplot as plt
###########################################################################################
# This is the code for Question2 part a
# In this part of code, using Pandas to read data from csv files
# This can contain the X-axis and Y-axis information which is useful to using pairplot
# Furthermore, do not forget to cut the previous 8 column data
#
# Source used in part a
# How to use pandas to read csv
# Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
# How to use iloc(pandas) to getting the 8 previous col data
# Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
# How to use seaborn to draw a pairplot
# Source: https://seaborn.pydata.org/generated/seaborn.pairplot.html
###########################################################################################
csv_data = pd.read_csv("./hw01/data.csv")
# printing the result of reading data
# Checking the data reading
# print(csv_data)
# In the question2 a, only save the previous 8 column
pre_eight_column = csv_data.iloc[:, 0:8]
Y = csv_data.iloc[:, [8]]
# This is to check the slice operation of iloc
# print(pre_eight_column)

# If you want to check the figure, using the following code
sbn.pairplot(pre_eight_column)
# Getting the pairplot result
plt.show()
###########################################################################################
# This is the code for Question2 part b
#
# Source used in part b
# How to use np.mean to get the avg for row and col
# Source: https://www.geeksforgeeks.org/numpy-mean-in-python/
# How to use zeros_likt to create the same size matrix
# Source: https://www.geeksforgeeks.org/numpy-zeros_like-python/
# Source: https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
###########################################################################################
# print(type(pre_eight_column))
np_pre_eight_data = np.array(pre_eight_column)
# checking the result of getting the first 8 columns
# print(np_pre_eight_data)

average_list = np.mean(np_pre_eight_data, axis=0)
# print(average_list)
zero_mean_array = np.zeros_like(np_pre_eight_data)

# 0-mean function
# For each element in the same column, minus its average
row, col = np_pre_eight_data.shape
for j in range(col):
    current_avg = average_list[j]
    for i in range(row):
        zero_m_result = np_pre_eight_data[i, j] - current_avg
        zero_mean_array[i, j] = zero_m_result
# Checking the average result of 0-mean
# print(zero_mean_array)

# Getting the square matrix
# For each location in the matrix doing the square operation
square_mean_array = np.zeros_like(zero_mean_array)
for i in range(row):
    for j in range(col):
        # Using the square to get the square matrix
        square_mean_array[i, j] = np.square(zero_mean_array[i, j])
# print(square_mean_array)

square_mean_array_sum = np.sum(square_mean_array, axis=0)
# print(square_mean_array_sum)

square_mean_array_sqrt = [np.sqrt(item) for item in square_mean_array_sum]
# print(square_mean_array_sqrt)

# This part of code is to getting the rescaled dataset
num_sqrt = np.sqrt(38)
rescaled_dataset = np.zeros_like(zero_mean_array)
for j in range(col):
    current_sqrt_2b = square_mean_array_sqrt[j]
    for i in range(row):
        rescaled_dataset[i, j] = zero_mean_array[i, j] * \
            num_sqrt / current_sqrt_2b
print("This is the rescaled dataset of 2b")
print(rescaled_dataset)
print()

# This is to check the result of the rescaled dataset
square_test = np.zeros_like(rescaled_dataset)
for i in range(row):
    for j in range(col):
        # Using the square to get the square matrix
        square_test[i, j] = np.square(rescaled_dataset[i, j])
test_result_of_n = np.sum(square_test, axis=0)
# print(test_result_of_n)

print("Question 2b result:")
# Checking the result of question2 b
print(
    f"This is the result of checking(same with the number n) n: {test_result_of_n}")
print()
###########################################################################################
# This is the code for Question2 part c
#
# Source used in part c
# Using sklearn ridge to implement c
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
# How to get Coef_
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# How to draw line graph
# Source: https://jakevdp.github.io/PythonDataScienceHandbook/04.01-simple-line-plots.html
###########################################################################################
alpha = [0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 200, 300]
X = rescaled_dataset
Y = np.array(Y)
result = list()
for item in alpha:
    regression_q2 = Ridge(item)
    regression_q2.fit(X, Y)
    arr = regression_q2.coef_
    result.append(arr.tolist())
# Testing the result
# print(result)

puring_result = list()
for item in result:
    for i in item:
        puring_result.append(i)
# Testing the puring_result
# print(puring_result)

draw_y = list()
np_coefficient = np.array(puring_result)
# Testing the np_coefficient
# print(np_coefficient)

for i in range(col):
    draw_y.append(np_coefficient[:, i])
draw_y = np.array(draw_y)
# Testing the draw_y
# print(draw_y)

log_lamda = [np.log(item) for item in alpha]
# Testing the log_lamda
# print(log_lamda)

# to avoid the confused caused by the plt, just doing the annotation of these part of code
# If you want to check the result please cancel the annotation
# Draw 8 lines to check the result
# Notice here setting the marker to get the point on the graph
line1 = plt.plot(log_lamda, draw_y[0], 'red', marker='o', label='line1')
line2 = plt.plot(log_lamda, draw_y[1], 'brown', marker='o', label='line2')
line3 = plt.plot(log_lamda, draw_y[2], 'green', marker='o', label='line3')
line4 = plt.plot(log_lamda, draw_y[3], 'blue', marker='o', label='line4')
line5 = plt.plot(log_lamda, draw_y[4], 'orange', marker='o', label='line5')
line6 = plt.plot(log_lamda, draw_y[5], 'pink', marker='o', label='line6')
line7 = plt.plot(log_lamda, draw_y[6], 'purple', marker='o', label='line7')
line8 = plt.plot(log_lamda, draw_y[7], 'grey', marker='o', label='line8')
plt.xlabel('log(lambda)')
plt.ylabel('coefficient')
plt.legend()
plt.show()

###########################################################################################
# This is the code for Question2 part d
#
# Source used in part d
# How to use np.copy(hardcopy different with view)
# Source: https://numpy.org/doc/stable/reference/generated/numpy.copy.html
# How to use the Ridge do the predict
# Source: https://machinelearningmastery.com/ridge-regression-with-python/
###########################################################################################
# print(row)
Error_list = list()
lambda_list = list()
for lamda in range(0, 501, 1):
    # print(i/10)
    current_lamda = lamda/10
    lambda_list.append(current_lamda)
    Error = 0
    # LOOCV method
    for pointer in range(0, row):
        # Getting the matrix
        temp_X = np.copy(rescaled_dataset)
        temp_Y = np.copy(Y)
        # Getting the test data with the pointer
        Testing_X = temp_X[pointer]
        Testing_Y = temp_Y[pointer]
        # Removing the data from the array
        Training_X = np.delete(temp_X, pointer, 0)
        Training_Y = np.delete(temp_Y, pointer, 0)

        # Start training the model by using Ridge
        # Using the lambda from 0-50 to set Ridge
        reg_q2_d = Ridge(current_lamda)
        # Training
        reg_q2_d.fit(Training_X, Training_Y)
        # Notice here adding [] avoiding the shape error
        predict_y = reg_q2_d.predict([Testing_X])
        # Getting the error
        error = np.square(predict_y-Testing_Y)
        # print(error)
        Error += error
    # Adding the error to the list
    Error_list.append(Error/row)

# Rescale the error list [[[]]]
rescaled_Error_list = list()
for item in Error_list:
    for itm in item:
        for i in itm:
            # Store the data into a list
            rescaled_Error_list.append(i)
# Checking the result of resizing the list
# print(rescaled_Error_list)

# Getting the min and max error
min_error_value = min(rescaled_Error_list)
max_error_value = max(rescaled_Error_list)
# Getting the index
min_index = rescaled_Error_list.index(min_error_value)
max_index = rescaled_Error_list.index(max_error_value)
# getting the lambda
min_lambda = lambda_list[min_index]
max_lambda = lambda_list[max_index]

# Printing the result
print("Question 2d result:")
# printing the result of min and max error value
# The minimum error value: 1442.6982227952926. The current lambda value : 22.3
# The maximum error value: 1975.4147393421708. The current lambda value : 0.0
print(
    f'The minimum error value of Ridge: {min_error_value}. The current lambda value of Ridge: {min_lambda}')
print(
    f'The maximum error value of Ridge: {max_error_value}. The current lambda value of Ridge: {max_lambda}')

# These part of code is to printing the plot of question 2d
Question_2d_plot = plt.plot(
    lambda_list, rescaled_Error_list, 'red', label='line')
plt.xlabel('lambda range')
plt.ylabel('Error average')
plt.legend()
plt.show()

# This is to compare with the standard linear regression
# Notice: Here is to create a standard linear regression function
std_regression = LinearRegression()
# Using the X Y to do the training
model = std_regression.fit(X, Y)
# Same way in predicting
predict_std_y = model.predict(X)
# Getting the difference and then doing the average
total_difference = 0
for i in range(row):
    current_square_difference = np.square(Y[i]-predict_std_y[i])
    total_difference += current_square_difference

avg_difference = total_difference/row
# traditionally better than the previous max_error_value 1085<1442
# result [1085.8364079] This result is to compare with the LOOCV result
# Better than LOOCV
print(f"standard linear regression: {avg_difference}")
print()
###########################################################################################
# This is the code for Question2 part e
# This part of code is similar to part c
#
# How to use Lasso
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
###########################################################################################
# This list is from the instruction of hw1
alpha = [0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 200, 300]
result_lasso = list()
for item in alpha:
    # Creating the Lasso model
    regression_lasso = Lasso(item)
    # Same way in fitting and training
    regression_lasso.fit(X, Y)
    lasso_coefficient = regression_lasso.coef_
    # Notice the shape of the result
    result_lasso.append(lasso_coefficient.tolist())
# print(result_lasso)

np_coefficient_lasso = np.array(result_lasso)
# print(np_coefficient_lasso)

draw_y_lasso = list()
for i in range(col):
    draw_y_lasso.append(np_coefficient_lasso[:, i])
np_draw_y_lasso = np.array(draw_y_lasso)
# print(np_draw_y_lasso)

# Here is similar to question c
# Notice using np.log to get the log result(X-axsis)
log_lamda_lasso = [np.log(item) for item in alpha]

# Draw 8 lines to check the result
# Notice here setting the marker to get the point on the graph
line1_lasso = plt.plot(
    log_lamda_lasso, np_draw_y_lasso[0], 'red', marker='o', label='line1_lasso')
line2_lasso = plt.plot(
    log_lamda_lasso, np_draw_y_lasso[1], 'brown', marker='o', label='line2_lasso')
line3_lasso = plt.plot(
    log_lamda_lasso, np_draw_y_lasso[2], 'green', marker='o', label='line3_lasso')
line4_lasso = plt.plot(
    log_lamda_lasso, np_draw_y_lasso[3], 'blue', marker='o', label='line4_lasso')
line5_lasso = plt.plot(
    log_lamda_lasso, np_draw_y_lasso[4], 'orange', marker='o', label='line5_lasso')
line6_lasso = plt.plot(
    log_lamda_lasso, np_draw_y_lasso[5], 'pink', marker='o', label='line6_lasso')
line7_lasso = plt.plot(
    log_lamda_lasso, np_draw_y_lasso[6], 'purple', marker='o', label='line7_lasso')
line8_lasso = plt.plot(
    log_lamda_lasso, np_draw_y_lasso[7], 'grey', marker='o', label='line8_lasso')
plt.xlabel('log(lambda)')
plt.ylabel('coefficient')
plt.legend(fontsize=8)
plt.show()

###########################################################################################
# This is the code for Question2 part f
# This part of code is similar to part d
###########################################################################################
Error_list_lasso = list()
lambda_list_lasso = list()
# Notice here the question is setting from 0 to 20
for lamda_lasso in range(0, 201, 1):
    # print(i/10)
    current_lamda_lasso = lamda_lasso/10
    lambda_list_lasso.append(current_lamda_lasso)
    Error_lasso = 0
    # LOOCV method
    for pointer_lasso in range(0, row):
        # Copy the matrix
        temp_X_lasso = np.copy(rescaled_dataset)
        temp_Y_lasso = np.copy(Y)
        # Getting the test data
        Testing_X_lasso = temp_X_lasso[pointer_lasso]
        Testing_Y_lasso = temp_Y_lasso[pointer_lasso]
        # Removing the test data
        Training_X_lasso = np.delete(temp_X_lasso, pointer_lasso, 0)
        Training_Y_lasso = np.delete(temp_Y_lasso, pointer_lasso, 0)

        # Setting the Lasso model
        reg_q2f_lasso = Lasso(current_lamda_lasso)
        # Training the model
        reg_q2f_lasso.fit(Training_X_lasso, Training_Y_lasso)
        # Doing the prediction
        predict_y_lasso = reg_q2f_lasso.predict([Testing_X_lasso])
        # Calculating the error
        error_lasso = np.square(predict_y_lasso-Testing_Y_lasso)
        # print(error)
        Error_lasso += error_lasso
    # Getting the average error of lasso -> list
    Error_list_lasso.append(Error_lasso/row)
# print(Error_list_lasso)

# here is to change the type of the result [[]]
# Puring the result for the following figure plot
puring_error_lasso = list()
for item in Error_list_lasso:
    for i in item:
        # Store the data into list for lasso result
        puring_error_lasso.append(i)
# print(puring_error_lasso)
# print(len(puring_error_lasso))

# Getting the min and max error value of lasso
min_error_value_lasso = min(puring_error_lasso)
max_error_value_lasso = max(puring_error_lasso)
# Getting the index of value
min_index_lasso = puring_error_lasso.index(min_error_value_lasso)
max_index_lasso = puring_error_lasso.index(max_error_value_lasso)
# Getting the corresponding lambda
min_lambda_lasso = lambda_list_lasso[min_index_lasso]
max_lambda_lasso = lambda_list_lasso[max_index_lasso]

print()
print("Question 2f result:")
# The minimum error value of lasso: 1586.6715081806428. The current lambda value of lasso : 5.5
# The maximum error value of lasso: 1973.8286526002037. The current lambda value of lasso : 0.0
print(
    f'The minimum error value of lasso: {min_error_value_lasso}. The current lambda value of lasso : {min_lambda_lasso}')
print(
    f'The maximum error value of lasso: {max_error_value_lasso}. The current lambda value of lasso : {max_lambda_lasso}')
print()

# These part of code is to printing the plot of question 2f
Question_2f_plot = plt.plot(
    lambda_list_lasso, puring_error_lasso, 'red', label='line_lasso')
plt.xlabel('lambda range(lasso)')
plt.ylabel('Error average(lasso)')
plt.legend(loc='best')
plt.show()
