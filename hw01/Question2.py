from re import X
import numpy as np
import seaborn as sbn
import pandas as pd
import matplotlib.pyplot as plt

###########################################################################################
# This is the code for Question2 part a
# In this part of code, using Pandas to read data from csv files
# This can contain the X-axis and Y-axis information which is useful to using pairplot
# Furthermore, do not forget to cut the previous 8 column data
###########################################################################################
csv_data = pd.read_csv("./hw01/data.csv")
# printing the result of reading data
# Checking the data reading
# print(csv_data)
# In the question2 a, only save the previous 8 column
pre_eight_column = csv_data.iloc[:,0:8]
Y = csv_data.iloc[:,[8]]
# This is to check the slice operation of iloc
# print(pre_eight_column)

# If you want to check the figure, using the following code
# sbn.pairplot(pre_eight_column)
# Getting the pairplot result
#plt.show()
###########################################################################################
# This is the code for Question2 part b
###########################################################################################
# print(type(pre_eight_column))
np_pre_eight_data = np.array(pre_eight_column)
# checking the result of getting the first 8 columns
# print(np_pre_eight_data)

average_list = np.mean(np_pre_eight_data,axis=0)
# print(average_list)
zero_mean_array = np.zeros_like(np_pre_eight_data)

# 0 mean function
row,col = np_pre_eight_data.shape
for j in range(col):
    current_avg = average_list[j]
    for i in range(row):
        zero_m_result = np_pre_eight_data[i,j] - current_avg
        zero_mean_array[i,j] = zero_m_result

# Checking the average result
# print(zero_mean_array)

# Getting the square matrix
square_mean_array = np.zeros_like(zero_mean_array)
for i in range(row):
    for j in range(col):
        square_mean_array[i,j] = np.square(zero_mean_array[i,j])
# print(square_mean_array)
square_average_list = np.mean(square_mean_array,axis=0)
# check the result
# print(square_average_list)
sqrt_square_average_list = [np.sqrt(item) for item in square_average_list]
# check the result
# print(sqrt_square_average_list)

row,col = np_pre_eight_data.shape
rescaled_dataset = np.zeros_like(np_pre_eight_data)
for j in range(col):
    sqrt_current_avg = sqrt_square_average_list[j]
    for i in range(row):
        division_by_sqrt = np_pre_eight_data[i,j] / sqrt_current_avg
        rescaled_dataset[i,j] = division_by_sqrt

# Checking the result of question2 b
# print(rescaled_dataset)

###########################################################################################
# This is the code for Question2 part c
# Using sklearn ridge to implement c
###########################################################################################
from sklearn.linear_model import Ridge
alpha = [0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 200, 300]
X = rescaled_dataset
Y = np.array(Y)
result = list()
for item in alpha:
    regression_q2 = Ridge(item)
    regression_q2.fit(X,Y)
    arr = regression_q2.coef_
    result.append(arr.tolist())
#print(result)

puring_result = list()
for item in result:
    for i in item:
        puring_result.append(i)
#print(puring_result)

draw_y = list()
np_coefficient = np.array(puring_result)
# print(np_coefficient)

for i in range(col):
   draw_y.append(np_coefficient[:,i])
draw_y = np.array(draw_y)
# print(draw_y)

log_lamda = [np.log(item) for item in alpha]
# print(log_lamda)

# Draw 8 lines to check the result
# line1=plt.plot(log_lamda,draw_y[0],'red',label='line1')
# line2=plt.plot(log_lamda,draw_y[1],'brown',label='line2')
# line3=plt.plot(log_lamda,draw_y[2],'green',label='line3')
# line4=plt.plot(log_lamda,draw_y[3],'blue',label='line4')
# line5=plt.plot(log_lamda,draw_y[4],'orange',label='line5')
# line6=plt.plot(log_lamda,draw_y[5],'pink',label='line6')
# line7=plt.plot(log_lamda,draw_y[6],'purple',label='line7')
# line8=plt.plot(log_lamda,draw_y[7],'grey',label='line8')
# plt.xlabel('log(lambda)')
# plt.ylabel('coefficient')
# plt.legend()
# plt.show()

###########################################################################################
# This is the code for Question2 part d
###########################################################################################
# print(row)
Error_list = list()
lambda_list = list()
for lamda in range(0,501,1):
    #print(i/10)
    current_lamda = lamda/10
    lambda_list.append(current_lamda)
    Error = 0
    for pointer in range(0,row):
        temp_X = np.copy(rescaled_dataset)
        temp_Y = np.copy(Y)
        Testing_X = temp_X[pointer]
        Testing_Y = temp_Y[pointer]
        Training_X = np.delete(temp_X,pointer,0)
        Training_Y = np.delete(temp_Y,pointer,0)

        
        reg_q2_d = Ridge(current_lamda)
        reg_q2_d.fit(Training_X,Training_Y)
        predict_y = reg_q2_d.predict([Testing_X])
        error = np.square(predict_y-Testing_Y)
        # print(error)
        Error += error
    Error_list.append(Error/row)

rescaled_Error_list = list()
for item in Error_list:
    for itm in item:
        for i in itm:
            rescaled_Error_list.append(i)
# Checking the result of resizing the list
# print(rescaled_Error_list)

min_error_value = min(rescaled_Error_list)
max_error_value = max(rescaled_Error_list)
min_index = rescaled_Error_list.index(min_error_value)
max_index = rescaled_Error_list.index(max_error_value)
min_lambda = lambda_list[min_index]
max_lambda = lambda_list[max_index]

# printing the result of min and max error value
# print(f'The minimum error value: {min_error_value}. The current lambda value : {min_lambda}')
# print(f'The maximum error value: {max_error_value}. The current lambda value : {max_lambda}')

# These part of code is to printing the plot of question 2d
# Question_2d_plot=plt.plot(lambda_list,rescaled_Error_list,'red',label='line')
# plt.xlabel('lambda range')
# plt.ylabel('Error average')
# plt.legend()
# plt.show()

# This is to compare with the standard linear regression
from sklearn.linear_model import LinearRegression
std_regression = LinearRegression()
model = std_regression.fit(X,Y)
predict_std_y = model.predict(X)
total_difference = 0
for i in range(row):
    current_square_difference = np.square(Y[i]-predict_std_y[i])
    total_difference += current_square_difference

avg_difference = total_difference/row
# traditionally better than the previous max_error_value 1085<1442
# print(avg_difference)

###########################################################################################
# This is the code for Question2 part e
###########################################################################################
from sklearn.linear_model import Lasso
alpha = [0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 200, 300]
result_lasso = list()
for item in alpha:
    regression_lasso = Lasso(item)
    regression_lasso.fit(X,Y)
    lasso_coefficient = regression_lasso.coef_
    result_lasso.append(lasso_coefficient.tolist())
#print(result_lasso)

np_coefficient_lasso = np.array(result_lasso)
print(np_coefficient_lasso)

draw_y_lasso = list()
for i in range(col):
   draw_y_lasso.append(np_coefficient_lasso[:,i])
np_draw_y_lasso = np.array(draw_y_lasso)
#print(np_draw_y_lasso)

log_lamda_lasso = [np.log(item) for item in alpha]

# Draw 8 lines to check the result
# line1_lasso=plt.plot(log_lamda_lasso,np_draw_y_lasso[0],'red',label='line1_lasso')
# line2_lasso=plt.plot(log_lamda_lasso,np_draw_y_lasso[1],'brown',label='line2_lasso')
# line3_lasso=plt.plot(log_lamda_lasso,np_draw_y_lasso[2],'green',label='line3_lasso')
# line4_lasso=plt.plot(log_lamda_lasso,np_draw_y_lasso[3],'blue',label='line4_lasso')
# line5_lasso=plt.plot(log_lamda_lasso,np_draw_y_lasso[4],'orange',label='line5_lasso')
# line6_lasso=plt.plot(log_lamda_lasso,np_draw_y_lasso[5],'pink',label='line6_lasso')
# line7_lasso=plt.plot(log_lamda_lasso,np_draw_y_lasso[6],'purple',label='line7_lasso')
# line8_lasso=plt.plot(log_lamda_lasso,np_draw_y_lasso[7],'grey',label='line8_lasso')
# plt.xlabel('log(lambda)')
# plt.ylabel('coefficient')
# plt.legend()
# plt.show()

###########################################################################################
# This is the code for Question2 part f
###########################################################################################
Error_list_lasso = list()
lambda_list_lasso = list()
for lamda_lasso in range(0,201,1):
    #print(i/10)
    current_lamda_lasso = lamda_lasso/10
    lambda_list_lasso.append(current_lamda_lasso)
    Error_lasso = 0
    for pointer_lasso in range(0,row):
        temp_X_lasso = np.copy(rescaled_dataset)
        temp_Y_lasso = np.copy(Y)
        Testing_X_lasso = temp_X_lasso[pointer_lasso]
        Testing_Y_lasso = temp_Y_lasso[pointer_lasso]
        Training_X_lasso = np.delete(temp_X_lasso,pointer_lasso,0)
        Training_Y_lasso = np.delete(temp_Y_lasso,pointer_lasso,0)

