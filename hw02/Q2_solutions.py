#####################################################################

#####################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import jax.numpy as jnp
from jax import grad, random
import jax as jax
from scipy.optimize import minimize
import tqdm as tqdm
#####################################################################
# Question2 a
#####################################################################
# From this question we can define A and b in the first step
A = np.array([[1, 0, 1, -1], [-1, 1, 0, 2], [0, -1, -2, 1]])
b = np.array([[1], [2], [3]])
x_k = np.array([[1], [1], [1], [1]])
learning_rate = 0.1

result_dict = {0: x_k}
result_list = [0]

norm_list = list()
for index in range(1, 9999):
    current_gradients = np.dot(A.T, (np.dot(A, x_k)-b))
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
# for i in range(abs(len(result_dict) - len(norm_list))):
#     del result_dict[len(norm_list)-i]

print("Here is the answer of question2 (a):")
for i in range(5):
    x_result = result_dict[i].tolist()
    x_result = [iii for item in x_result for iii in item]
    print(f"k={i}, x(k={i})={list(x_result)}")

# notice here using norm list to limit the result_dict
for i in range(len(result_dict)-1, len(result_dict)-6, -1):
    x_result = result_dict[i].tolist()
    x_result = [iii for item in x_result for iii in item]
    print(f"k={i}, x(k={i})={list(x_result)}")
print()
#####################################################################
# Question2 b
# From this question we can define A and b in the first step
# We can get the following ((A∇f(x^((k) ) ))^T (Ax^((k) )-b)+((Ax^((k) )-b))^T A∇f(x^((k) ) ))/(〖2(A∇f(x^((k) ) ))〗^T A∇f(x^((k) ) ) )
#####################################################################
A = np.array([[1, 0, 1, -1], [-1, 1, 0, 2], [0, -1, -2, 1]])
b = np.array([[1], [2], [3]])
x_k_b = np.array([[1], [1], [1], [1]])



def learning_rate_function(x_k_b):
    A_x_k_b = np.dot(A,x_k_b)-b
    A_f_x_k = np.dot(A,np.dot(A.T,A_x_k_b))
    Top_line = np.dot((A_f_x_k.T),A_x_k_b) + np.dot((A_x_k_b.T),A_f_x_k)
    Bottom_line = 2 * np.dot((A_f_x_k.T),A_f_x_k)
    learning_rate_result = Top_line/Bottom_line
    return learning_rate_result


# # print(current_learning_rate)
# # print(learning_rate_function(x_k_b))

learning_rate_lst = [0.1]
x_k_b_dict = {0:x_k_b}
norm_checker = list()

for index in range(9999):
    current_gradients_b = np.dot(A.T, (np.dot(A, x_k_b)-b))
    # print(f"gradients: {current_gradients_b}")
    # print(f"learning rate: {current_learning_rate}")
    # print(current_gradients_b)
    if index ==0:
        current_learning_rate = 0.1
    else:
        current_learning_rate = learning_rate_function(x_k_b)

    next_x_b = x_k_b - current_learning_rate * current_gradients_b
    # print(f"The next x: {next_x_b}")
    gradient_norm_b = np.linalg.norm(current_gradients_b)
    # print(f"norm: {gradient_norm_b}")
    # print(f"the norm value: {gradient_norm_b}")
    if gradient_norm_b < 0.001:
        break
    else:
        next_learning_rate = learning_rate_function(x_k_b)
        # print(f"learning rate: {next_learning_rate}")
        norm_checker.append(gradient_norm_b)
        x_k_b_dict.update({index:next_x_b})
        learning_rate_lst.append(next_learning_rate)
        current_learning_rate = next_learning_rate
        x_k_b = next_x_b


# index_array_d = np.array([i for i in range(len(learning_rate_lst))])
# ???
plt.plot()
plt.plot(index_array_d,learning_rate_lst)
plt.show()
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

# print("Here is the answer of question2 (d):")
# print(f"first row X train: {Train_X[0]}")
# print(f"last row X train: {Train_X[-1]}")
# print(f"first row X test: {Test_X[0]}")
# print(f"last row X test: {Test_X[-1]}")
# print(f"first row Y train: {Train_Y[0]}")
# print(f"last row Y train: {Train_Y[-1]}")
# print(f"first row Y test: {Test_Y[0]}")
# print(f"first row Y test: {Test_Y[-1]}")
# print()
#####################################################################
# Question2 e
#####################################################################
adding_one = np.array([[1] for i in range(len(Train_X))])
# print(adding_one.shape)
new_Train_X = np.hstack((adding_one, Train_X))
# inputs = jnp.array(Train_X)
# print(new_Train_X.shape)
# print(new_Train_X)
inputs = jnp.array(new_Train_X)
targets = jnp.array(Train_Y)
W = jnp.array([[1.0, 1.0, 1.0, 1.0]])
# print(W.shape)
# print(W.T.shape)

def predict(W):
  para_w_T = W.T
  predict_result = jnp.dot(inputs,para_w_T)
  return predict_result


def loss(W):
  preds = predict(W)
  return jnp.mean((jnp.sqrt(0.25*jnp.square(targets-preds)+1)-1))


predict_result = predict(W)
loss_result = loss(W)
# print(loss_result)

W_grad = grad(loss)(W)
# print(W_grad)

learning_rate = 1

loss_list = [loss_result]
weight_list = list()
previous_loss = loss_result
for index in range(99999):
    current_w = W -learning_rate * grad(loss)(W)
    current_loss = loss(current_w)
    # print(f"difference: {abs(previous_loss-current_loss)}")
    if abs(previous_loss-current_loss) < 0.0001:
        break
    else:
        loss_list.append(current_loss)
        previous_loss = current_loss
        weight_list.append(current_w)
        W = current_w

weight_array = jnp.array(weight_list)
index_list = [i for i in range(len(loss_list))]
train_loss = loss(weight_array[-1])
# print(train_loss)

def predict_test(W):
  para_w_T = W.T
  predict_result = jnp.dot(e_test_x,para_w_T)
  return predict_result

def loss_test(W):
  preds = predict_test(W)
  return jnp.mean((jnp.sqrt(0.25*jnp.square(e_test_y-preds)+1)-1))

adding_one_test = np.array([[1] for i in range(len(Test_X))])
new_test_x = np.hstack((adding_one_test, Test_X))
e_test_x = jnp.array(new_test_x)  
e_test_y = jnp.array(Test_Y)
test_loss = loss_test(weight_array[-1])

w_final = weight_array[-1]
# Using MAE to calculate the accuracy
def accuracy(targets,preds):
    return jnp.mean(jnp.abs(targets - preds))

train_acc = accuracy(targets,predict(w_final))
test_acc = accuracy(e_test_y, predict_test(w_final))


print()
print("Here is the answer of question2 (e):")
print(f"Iterration: {len(loss_list)-1}")
print(f"The final weight is: {weight_array[-1]}")
print(f"The Train loss is: {train_loss}")
print(f"The Test loss is: {test_loss}")
print(f"The Train accuracy is: {train_acc}")
print(f"The Test accuracy is: {test_acc}")


plt.plot(index_list, loss_list)
plt.show()
print()

# #####################################################################
# # Question2 f
# #####################################################################
A_f = np.array([[1, 0, 1, -1], [-1, 1, 0, 2], [0, -1, -2, 1]])
b_f = np.array([[1], [2], [3]])
x_k_b_f = np.array([[1], [1], [1], [1]])

w0 = jnp.array([[1.0, 1.0, 1.0, 1.0]])
optimal = minimize(learning_rate_function,x_k_b,method="BFGS")   
w_optimize = optimal
# print(w_optimize)

