#####################################################################

#####################################################################
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#####################################################################
# Question2 a
#####################################################################
# From this question we can define A and b in the first step
A = np.array([[1, 0, 1, -1], [-1, 1, 0, 2], [0, -1, -2, 1]])
b = np.array([[1], [2], [3]])
x_k = np.array([[1], [1], [1], [1]])
learning_rate = 0.1
A_T = A.transpose()
result_dict = {0: x_k}
result_list = [0]

norm_list = list()
for index in range(1, 9999):
    current_gradients = np.dot(A_T, (np.dot(A, x_k)-b))
    gradient_norm = np.linalg.norm(current_gradients)
    next_x = x_k-learning_rate*current_gradients
    if gradient_norm < 0.001:
        break
    # print(gradient_norm)
    if gradient_norm < 0.001:
        break
    else:
        norm_list.append(gradient_norm)
        result_dict.update({index: next_x})
        # test the list
        result_list.append(next_x)
        x_k = next_x

# Delete i times --> Using norm list to select corrent answer in the result_dict
for i in range(abs(len(result_dict) - len(norm_list))):
    del result_dict[len(norm_list)-i]

print("Here is the answer of question2 (a):")
for i in range(5):
    x_result = result_dict[i].tolist()
    x_result = [iii for item in x_result for iii in item]
    print(f"k={i}, x(k={i})={list(x_result)}")

# notice here using norm list to limit the result_dict
for i in range(len(result_dict)-1, len(result_dict)-6, -1):
    x_result = result_list[i].tolist()
    x_result = [iii for item in x_result for iii in item]
    print(f"k={i}, x(k={i})={list(x_result)}")
print()
#####################################################################
# Question2 b
#####################################################################
# From this question we can define A and b in the first step
A = np.array([[1, 0, 1, -1], [-1, 1, 0, 2], [0, -1, -2, 1]])
b = np.array([[1], [2], [3]])
x_k = np.array([[1], [1], [1], [1]])
learning_rate = 0.1
A_T = A.transpose()
result_dict = {0: x_k}
result_list = [0]

norm_list = list()
for index in range(1, 9999):
    current_gradients = np.dot(A_T, (np.dot(A, x_k)-b))
    gradient_norm = np.linalg.norm(current_gradients)
    next_x = x_k-learning_rate*current_gradients
    if gradient_norm < 0.001:
        break
    # print(gradient_norm)
    if gradient_norm < 0.001:
        break
    else:
        norm_list.append(gradient_norm)
        result_dict.update({index: next_x})
        # test the list
        result_list.append(next_x)
        x_k = next_x

# Delete i times --> Using norm list to select corrent answer in the result_dict
for i in range(abs(len(result_dict) - len(norm_list))):
    del result_dict[len(norm_list)-i]

# print("Here is the answer of question2 (a):")
# for i in range(5):
#     x_result = result_dict[i].tolist()
#     x_result = [iii for item in x_result for iii in item]
#     print(f"k={i}, x(k={i})={list(x_result)}")

# # notice here using norm list to limit the result_dict
# for i in range(len(result_dict)-1, len(result_dict)-6, -1):
#     x_result = result_list[i].tolist()
#     x_result = [iii for item in x_result for iii in item]
#     print(f"k={i}, x(k={i})={list(x_result)}")
print()
#####################################################################
# Question2 c
# Steepest is faster, value change
#####################################################################

#####################################################################
# Question2 d
#####################################################################
Original_Q2_data = pd.read_csv("./hw02/Q2.csv")
Original_train_X = Original_Q2_data.iloc[:, 1: 4]
Original_train_Y = Original_Q2_data.iloc[:, 6: 7]
Total_X_NAN = np.array(Original_train_X)
Total_Y_NAN = np.array(Original_train_Y)
Total_data_NAN = np.hstack((Total_X_NAN, Total_Y_NAN))
# How to remove NAN
# Source: https://note.nkmk.me/en/python-numpy-nan-remove/
puring_data = Total_data_NAN[~np.isnan(Total_data_NAN).any(axis=1)]

Total_X = puring_data[:, :3]
Total_Y = puring_data[:, 3:4]
# print(Total_X.shape)
# print(Total_Y.shape)

Q2_d_min_max_scalar = MinMaxScaler()
Q2_d_x_min_max = Q2_d_min_max_scalar.fit_transform(Total_X)


def get_half(length):
    if length % 2 == 0:
        half_index = length
    elif length % 2 == 1:
        half_index = length//2 + 1
    return half_index


half_index = get_half(len(Total_X))

# print(half_index)
Train_X = Q2_d_x_min_max[:half_index]
Test_X = Q2_d_x_min_max[half_index:len(Total_X)]
Train_Y = Total_Y[:half_index]
Test_Y = Total_Y[half_index:len(Total_Y)]

'''
• first row X train: [0.73059361, 0.00951267, 1.]
• last row X train: [0.87899543, 0.09926012, 0.3]
• first row X test: [0.26255708, 0.20677973, 0.1]
• last row X test: [0.14840183, 0.0103754, 0.9]
• first row Y train: 37.9
• last row Y train: 34.2
• first row Y test: 26.2
• last row Y test: 63.9
'''
print(f"first row X train: {Train_X[0]}")
print(f"last row X train: {Train_X[-1]}")
print(f"first row X test: {Test_X[0]}")
print(f"last row X test: {Test_X[-1]}")
print(f"first row Y train: {Train_Y[0]}")
print(f"last row Y train: {Train_Y[-1]}")
print(f"first row Y test: {Test_Y[0]}")
print(f"first row Y test: {Test_Y[-1]}")

#####################################################################
# Question2 e
#####################################################################

#####################################################################
# Question2 f
#####################################################################

#####################################################################
# Question2 g
#####################################################################
