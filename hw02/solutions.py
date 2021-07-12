#####################################################################


#####################################################################
import numpy as np
import pandas as pd
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
print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

# Greedy search is focused on train x and train y
