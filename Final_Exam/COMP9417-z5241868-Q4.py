import numpy as np
import pandas as pd  # not really needed, only for preference
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("./Final_Exam/Data/Q4_train.csv")
# print(data)
X = np.array(data.iloc[:, 1: 6])
# get the column contain y
y = np.array(data.iloc[:, 6: 7])
# print(X)
# print(y)
###########################################################
###########################################################
# Question 4 question c
###########################################################
###########################################################


def Loss_function_c(pred_y, y_truth):
    # Notice in the previous using sum instead of mean
    Difference_pred_truth = pred_y - y_truth
    Difference_square = np.square(Difference_pred_truth)
    current_loss = np.sum(Difference_square)
    return current_loss


def total_loss_c(X, y, Z, models):
    loss_sum = 0
    # Looping the model in the list
    for index in range(len(models)):
        # Finding the index where z equals to index
        index_checker = np.where(Z == index)
        data_X = X[index_checker]
        data_y = y[index_checker]
        X_shape_ = data_X.shape[0]
        if X_shape_ != 0:
            pass
        else:
            continue
        current_model = models[index]
        current_model_coef_ = current_model.coef_
        current_coef_T = current_model_coef_.T
        predict_y = data_X.dot(current_coef_T)
        # From this part is to compute the loss value
        loss_sum = loss_sum + Loss_function_c(predict_y, data_y)
    return loss_sum


# This part of code is from the instruction
# # Example, if M=1, we would just fit a single linear model
mod = LinearRegression().fit(X, y)
# all points would belong to a singlepartition.
Z = np.zeros(shape=X.shape[0])
# Wrapped the model in the list
model_list = [mod]
print(total_loss_c(X, y, Z, model_list))  # outputs 298.328178158043
print()
###########################################################
###########################################################
# Question 4 question e
###########################################################
###########################################################


def Loss_function(pred_y, y_truth):
    # Notice in the previous using sum instead of mean
    Difference_pred_truth = pred_y - y_truth
    Difference_square = np.square(Difference_pred_truth)
    current_loss = np.sum(Difference_square)
    return current_loss

# This part is similar to queestion4 c


def total_loss(X, y, Z, models):

    loss_sum = 0
    # Looping the model in the list
    for index in range(len(models)):
        # Finding the index where z equals to index
        index_checker = np.where(Z == index)
        data_X = X[index_checker]
        data_y = y[index_checker]
        X_shape_ = data_X.shape[0]
        if X_shape_ != 0:
            pass
        else:
            continue
        str_model_name = 'module_'+str(i)
        current_model = models[str_model_name]
        current_coef_ = current_model.coef_
        coef_T = current_coef_.T
        predict = data_X.dot(coef_T)
        loss_sum += Loss_function(predict, data_y)
    return loss_sum


# This part is  queestion4 d
def find_partitions(X, y, models):

    M = len(models)
    loss_array = np.zeros((M, X.shape[0]))
    for index in range(M):
        model_index = str(index)
        str_model_name = 'module_' + model_index
        current_model = models[str_model_name]
        model_coef = current_model.coef_
        model_coef_T = model_coef.T
        y_pred = X.dot(model_coef_T)
        square_difference = np.square(y_pred - y)
        loss_array[index] = square_difference
    Z = np.argmin(loss_array, axis=0)
    return Z


# This part is to loding the data into the model dict
# Notice plz change to your own data location
df_train = pd.read_csv("./Final_Exam/Data/Q4_train.csv", index_col=0)
X_train = np.array(df_train.iloc[:, :5])
Y_train = np.array(df_train.Y)
df_test = pd.read_csv("./Final_Exam/Data/Q4_test.csv", index_col=0)
X_test = np.array(df_test.iloc[:, :5])
Y_test = np.array(df_test.Y)
# This is to store the total loss of train and test
train_loss_list_all = list()
test_loss_list_all = list()

# Starting the proecessing createing the model dict and training
for m in range(30):
    M = m+1
    module_dict = dict()
    data_dict = dict()
    num_x, size_x = X_train.shape
    loss_array_size = (M, num_x)
    loss_array = np.zeros(loss_array_size)
    # init module
    # np.array([0,1,2,3,4,...,M])
    M_30_list = [i for i in range(M)]
    uint = np.array(M_30_list)
    X_train_length = X_train.shape[0]
    locator = X_train_length//M
    X_train_size = (locator)+1
    b_sample = np.tile(uint, X_train_size)[:X_train_length]
    # Cutting the data set and fit the data set to get the current model
    for i in range(M):
        sample_checker = np.where(b_sample == i)
        X_train_b = X_train[sample_checker]
        Y_train_b = Y_train[sample_checker]
        str_mdl_name = 'module_' + str(i)
        module_dict[str_mdl_name] = LinearRegression()
        current_dict_model = module_dict[str_mdl_name]
        current_dict_model.fit(X_train_b, Y_train_b)
    # train the module
    train_loss_list = list()
    test_loss_list = list()
    # Looping the j j is just a counter no accurate usage
    for j in range(25):
        # Finding the best value z which is the minimum of loss_array
        index = find_partitions(X_train, Y_train, module_dict)
        for i in range(M):
            # Getting the training x data
            index_checker = np.where(index == i)
            x_data_tag = 'xdata_' + str(i)
            data_dict[x_data_tag] = X_train[index_checker]
            X_dict_data = data_dict[x_data_tag]
            # Getting the training y data
            y_data_tag = 'ydata_' + str(i)
            data_dict[y_data_tag] = Y_train[index_checker]
            y_dict_data = data_dict[y_data_tag]
            X_train_len = X_train[index_checker].shape[0]
            # The same check as question4 c
            if X_train_len != 0:
                pass
            else:
                continue
            # Create the index for the model dict
            current_str_model_name = 'module_' + str(i)
            # Setting the LR model
            module_dict[current_str_model_name] = LinearRegression()
            # For each model do the training and fitting
            LR_model = module_dict[current_str_model_name]
            LR_model.fit(X_dict_data, y_dict_data)

        # For each turn collect the loss of each turn
        # Collect train loss
        total_loss_train = total_loss(
            X_train, Y_train, index, module_dict)
        train_loss_list.append(total_loss_train)

        # Collect the test loss
        index_test = find_partitions(X_test, Y_test, module_dict)
        total_loss_test = total_loss(
            X_test, Y_test, index_test, module_dict)
        test_loss_list.append(total_loss_test)
    # This is to check the data information
    # print(train_loss_list)

    # Here you can change the plot you want to check training and testing
    label_name = 'M='+str(M)
    plt.plot(train_loss_list, label=label_name, marker=',')
    plt.plot(test_loss_list, label=label_name, marker=',')
    # Collect the current loss into the total list
    train_loss_list_all.append(train_loss_list)
    test_loss_list_all.append(test_loss_list)
plt.legend(loc='best')
plt.show()
