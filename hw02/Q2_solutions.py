#####################################################################

#####################################################################
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    if gradient_norm < 0.001:
        break
    # print(gradient_norm)

    if gradient_norm < 0.001:
        break
    else:
        norm_list.append(gradient_norm)
        next_x = x_k-learning_rate*current_gradients
        result_dict.update({index: next_x})
        # test the list
        result_list.append(next_x)
        x_k = next_x

# Delete i times --> Using norm list to select corrent answer in the result_dict
for i in range(abs(len(result_dict) - len(norm_list))):
    del result_dict[len(norm_list)-i]

print("Here is the answer of question2 (a_:")
for i in range(5):
    x_result = result_dict[i].tolist()
    x_result = [round(iii, 3) for item in x_result for iii in item]
    print(f"k={i}, x(k={i})={tuple(x_result)}")

# notice here using norm list to limit the result_dict
for i in range(len(result_dict)-1, len(result_dict)-6, -1):
    x_result = result_list[i].tolist()
    x_result = [round(iii, 3) for item in x_result for iii in item]
    print(f"k={i}, x(k={i})={tuple(x_result)}")
print()
#####################################################################
# Question2 b
#####################################################################

#####################################################################
# Question2 c
#####################################################################

#####################################################################
# Question2 d
#####################################################################
