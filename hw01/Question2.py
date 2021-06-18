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
print(np_coefficient)

for i in range(col):
   draw_y.append(np_coefficient[:,i])
draw_y = np.array(draw_y)
print(draw_y)

log_lamda = [np.log(item) for item in alpha]
print(log_lamda)

line1=plt.plot(log_lamda,draw_y[0],'r--',label='type1')
line2=plt.plot(log_lamda,draw_y[1],'g--',label='type2')
line3=plt.plot(log_lamda,draw_y[2],'b--',label='type3')
line4=plt.plot(log_lamda,draw_y[3],'p--',label='type4')
line5=plt.plot(log_lamda,draw_y[4],'c--',label='type5')
line6=plt.plot(log_lamda,draw_y[5],'m--',label='type6')
line7=plt.plot(log_lamda,draw_y[6],'y--',label='type7')
line8=plt.plot(log_lamda,draw_y[7],'k--',label='type8')

plt.xlabel('log(lamda)')
plt.ylabel('coefficient')
plt.legend()
plt.show()

