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
# This is to check the slice operation of iloc
print(pre_eight_column)
sbn.pairplot(pre_eight_column)
# Getting the pairplot result
#plt.show()
###########################################################################################
# This is the code for Question2 part b
###########################################################################################
print(type(pre_eight_column))
np_pre_eight_data = np.array(pre_eight_column)
print(np_pre_eight_data)

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
print(square_average_list)
sqrt_square_average_list = [np.sqrt(item) for item in square_average_list]
print(sqrt_square_average_list)

row,col = np_pre_eight_data.shape
rescaled_dataset = np.zeros_like(np_pre_eight_data)
for j in range(col):
    sqrt_current_avg = sqrt_square_average_list[j]
    for i in range(row):
        division_by_sqrt = np_pre_eight_data[i,j] / sqrt_current_avg
        rescaled_dataset[i,j] = division_by_sqrt

print(rescaled_dataset)

###########################################################################################
# This is the code for Question2 part c
###########################################################################################