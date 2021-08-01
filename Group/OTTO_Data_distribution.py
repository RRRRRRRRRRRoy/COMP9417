##############################################################################################################################
##############################################################################################################################
# This file is for COMP9417 Group project
# Using Otto Product dataset to do the training and testing
# Data Download link
# Source: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
#
# This is a old question you can also find some information for other guys resources
# Source: https://github.com/wepe/Kaggle-Solution/tree/master/Otto%20Group%20Product%20Classification%20Challenge
# Source: https://github.com/amaltarghi/Otto-Group-Product-Classification-Challenge
# Source: https://github.com/zhouhaozeng/kaggle-otto-classification
##############################################################################################################################
##############################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# Using pandas reading data
# Here change the location to your data location
# Downloading from Kaggle
# Source: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
raw_train_data = pd.read_csv("D:\\train.csv")
raw_test_data = pd.read_csv("D:\\test.csv")

# using iloc provided by pandas to cutting the data into training set and test set
# SOurce: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
Train_data = raw_train_data.iloc[:, 1:95]
labels = set(Train_data.iloc[:, 93:94].target)
Valid_x_pd = raw_test_data.iloc[:, 1:94]
Valid_x = np.array(Valid_x_pd)
Total_x = np.array(Train_data.iloc[:, 0:93])
Total_y = np.array(Train_data.iloc[:, 93:94])
# print(Total_y)

# Using the describe function provided by pandas can get the information
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
Train_data_info = Train_data.describe()
valid_data_info = Valid_x_pd.describe()
print("The description of Training dataset")
print(Train_data_info)
print("The description of Testing dataset")
print(valid_data_info)

# Do not forget to use set() to delete the same information
print()
print(f"The size of Train_X: {Total_x.shape}")
print(f"The size of Train_Y: {Total_y.shape}")
print(f"The size of Test_X: {Valid_x.shape}")
print(
    f"Is there Nan in the training X? {list(set(np.isnan(Total_x).any(axis=1)))}")
print(
    f"Is there Nan in the testing data set? {list(set(np.isnan(Valid_x).any(axis=1)))}")
print(
    f"The number of current labels: {len(set([item for i in Total_y.tolist() for item in i ]))}")
print(
    f"The current labels: {set([item for i in Total_y.tolist() for item in i ])}")
print()

# Setting up a dictionary to counting the feature in the dictionary
Lable_counter = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                 "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}
# Looping the data in the current training dataset
for item in Total_y:
    Lable_counter[item[-1]] += 1
# print(Lable_counter)

# Printing the result image to check the current data distribution
plt.bar(Lable_counter.keys(), Lable_counter.values(), width=0.45)
plt.xlabel("Classes", fontsize=10)
plt.ylabel("Number of Items", fontsize=10)
plt.savefig("./class_feature_relationship.jpg")
plt.show()

# For each feature counting the total number and compare the relationship
sum_result = Total_x.sum(axis=0)
feature_dict = dict()
for index in range(len(sum_result)):
    feature_dict.update({index+1: sum_result[index]})

# Printing the result image to check the current data distribution
# How to draw a horizontal bar plot
# Source: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.barh.html
index_list = [f"F{i}" for i in feature_dict.keys()]
feature_number = [i for i in feature_dict.values()]
plt.figure(figsize=(15, 20))
plt.xlabel("Sum of Features", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.barh(index_list, feature_number)
plt.savefig('./feature_relationship.jpg')
plt.show()
