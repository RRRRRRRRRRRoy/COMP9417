import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


Total_data = pd.read_csv("./Final_Exam/Data/Q1.csv")
X = np.array(Total_data.iloc[:, 1: 31])
# get the column contain y
y = np.array(Total_data.iloc[:, 31: 32])
# print("The result of Data X:")
# print(X)
# print("The result of Data y:")
# print(y)
print(f"The shape of Data X: {X.shape}")
print(f"The shape of Data y: {y.shape}")


####################################################################################################################################
####################################################################################################################################
# Question 1 a
####################################################################################################################################
####################################################################################################################################
coefficient_list = list()
np.random.seed(12)
for item in range(500):
    # generating train-i
    # i random list --> range(0-499) len(500) !!! important !!!
    random_list = np.random.randint(0, 250, 250)
    boostrap_train_X = np.zeros_like(X)
    boostrap_train_Y = np.zeros_like(y)
    for index in range(250):
        boostrap_temp_X = X[random_list[index]]
        boostrap_temp_Y = y[random_list[index]]
        boostrap_train_X[index] = boostrap_temp_X
        boostrap_train_Y[index] = boostrap_temp_Y

    # Notice: Here is the same para in Question C
    classifier_d = LogisticRegression(
        C=1000, solver='liblinear', penalty='l1')

    # Getting the parameters
    classifier_d.fit(boostrap_train_X, boostrap_train_Y.ravel())
    coefficient_result = classifier_d.coef_
    coefficient_list.append(coefficient_result)

# Purify the structure of the list
purify_coeffcient_list = list()
for item in coefficient_list:
    for i in item:
        purify_coeffcient_list.append(i)

# Getting the previous 90% lines
coefficient_list_90 = np.array(purify_coeffcient_list[:450])
# print(coefficient_list_9000.shape)
# print(type(coefficient_list_90))

# Getting the 50%
# fifty_column = np.percentile(purify_coeffcient_list, 50, axis=0)
# The question Said get mean for each bar
# Therefore using 9000 list to get the mean
mean_column = np.mean(purify_coeffcient_list, axis=0)
# # Getting the 5%
fifth_column_1a = np.percentile(coefficient_list_90, 5, axis=0)
# # Getting the 95%
ninety_fifth_column_1a = np.percentile(coefficient_list_90, 95, axis=0)

lower_1a = fifth_column_1a
upper_1a = ninety_fifth_column_1a

# Generating the data list to draw the plot
bar_data_1a = list()
for index in range(len(fifth_column_1a)):
    bar_data_1a.append([fifth_column_1a[index], ninety_fifth_column_1a[index]])


# Checking the color of each bar
# using the absolute length to draw the bar plot
# Checking the span of each bar and print the color
for index in range(len(bar_data_1a)):
    ndarray_data = np.array(bar_data_1a[index])
    # This is the red bar
    # Check the bottom and the height of each bar
    if fifth_column_1a[index] <= 0 and ninety_fifth_column_1a[index] >= 0:
        plot_draw = plt.bar(
            index, bottom=fifth_column_1a[index], height=ninety_fifth_column_1a[index]+abs(fifth_column_1a[index]), color='red')
    else:
        # This is the blue bar
        if fifth_column_1a[index] < 0 and ninety_fifth_column_1a[index] < 0:
            plot_draw = plt.bar(
                index, bottom=ninety_fifth_column_1a[index], height=fifth_column_1a[index]-ninety_fifth_column_1a[index], color='blue')
        if fifth_column_1a[index] > 0 and ninety_fifth_column_1a[index] > 0:
            plot_draw = plt.bar(
                index, bottom=fifth_column_1a[index], height=ninety_fifth_column_1a[index]-fifth_column_1a[index], color='blue')
label_XXX = [i for i in range(len(mean_column))]

# Draw the mean dot on the graph
plt.plot(label_XXX,
         mean_column, "o", color="lime", label="Mean Point", markersize="4")
plt.title('Question1 (a)', color='black')
plt.legend()
plt.savefig("./Final_Exam/Q1_a.png")
plt.show()


####################################################################################################################################
####################################################################################################################################
# Question 1 b
####################################################################################################################################
####################################################################################################################################
len_X, size_p = X.shape
print(
    f"The length of the train_X is {len_X}, the size of each Train_X is {size_p}")
np.random.seed(20)
Bootstrap_size = 500


logisticRegression_1b_full_model = LogisticRegression(
    C=1000, solver='liblinear', penalty='l1')
logisticRegression_1b_full_model.fit(X, y.ravel())
# THis is the beta
beta = logisticRegression_1b_full_model.coef_
# This is the beta_0
beta_0 = logisticRegression_1b_full_model.intercept_

# These 2 parts are based on the instruction
Numerator_part_p_i = np.exp(beta_0+X.dot(beta.T))
denominator_part_p_i = 1 + np.exp(beta_0+X.dot(beta.T))
equation_p_i = Numerator_part_p_i / denominator_part_p_i

# This list is to store the coef data
coefficient_list_1b = list()
# This list is to store the data y
new_y_data = list()

for index in range(Bootstrap_size):
    Bernoulli_result = np.random.binomial(n=1, p=equation_p_i).squeeze(1)
    new_y_data.append(np.array([Bernoulli_result]))

for index in range(Bootstrap_size):
    # The training data X didn't change
    random_list = np.random.randint(0, 250, 250)
    X_4_1b = np.zeros_like(X)
    y_4_1b = np.zeros_like(np.array(new_y_data))
    for index in range(250):
        boostrap_temp_X = X[random_list[index]]
        boostrap_temp_Y = new_y_data[random_list[index]]
        X_4_1b[index] = boostrap_temp_X
        y_4_1b[index] = boostrap_temp_Y

    model_4_1b = LogisticRegression(
        C=1000, solver='liblinear', penalty='l1')
    model_4_1b.fit(X_4_1b, y_4_1b.ravel())
    current_coef = model_4_1b.coef_
    coefficient_list_1b.append(current_coef)

print(np.array(coefficient_list_1b).shape)
# Purify the structure of the list
purify_coeffcient_list_1b = list()
for item in coefficient_list_1b:
    for i in item:
        purify_coeffcient_list_1b.append(i)
print(np.array(purify_coeffcient_list_1b).shape)


# Getting the previous 90% lines
coefficient_list_90 = np.array(purify_coeffcient_list_1b[:450])
# print(coefficient_list_9000.shape)
# print(type(coefficient_list_90))

# Getting the 50%
# fifty_column = np.percentile(purify_coeffcient_list, 50, axis=0)
# The question Said get mean for each bar
# Therefore using 9000 list to get the mean
mean_column = np.mean(purify_coeffcient_list_1b, axis=0)
# # Getting the 5%
fifth_column_1b = np.percentile(coefficient_list_90, 5, axis=0)
# # Getting the 95%
ninety_fifth_column_1b = np.percentile(coefficient_list_90, 95, axis=0)

lower_1b = fifth_column_1b
upper_1b = ninety_fifth_column_1b

# Generating the data list to draw the plot
bar_data_1b = list()
for index in range(len(fifth_column_1b)):
    bar_data_1b.append([fifth_column_1b[index], ninety_fifth_column_1b[index]])

color_list = list()
for index in range(size_p):
    if fifth_column_1b[index] <= 0 and ninety_fifth_column_1b[index] >= 0:
        color_list.append('red')
    else:
        color_list.append('blue')

# Generating the data list to draw the plot
bar_data_1b = list()
for index in range(len(fifth_column_1b)):
    bar_data_1b.append([fifth_column_1b[index], ninety_fifth_column_1b[index]])


# Checking the color of each bar
# using the absolute length to draw the bar plot
# Checking the span of each bar and print the color
for index in range(len(bar_data_1b)):
    ndarray_data = np.array(bar_data_1b[index])
    # This is the red bar
    # Check the bottom and the height of each bar
    if fifth_column_1b[index] <= 0 and ninety_fifth_column_1b[index] >= 0:
        plot_draw = plt.bar(
            index, bottom=fifth_column_1b[index], height=ninety_fifth_column_1b[index]+abs(fifth_column_1b[index]), color='red')
    else:
        # This is the blue bar
        if fifth_column_1b[index] < 0 and ninety_fifth_column_1b[index] < 0:
            plot_draw = plt.bar(
                index, bottom=ninety_fifth_column_1b[index], height=fifth_column_1b[index]-ninety_fifth_column_1b[index], color='blue')
        if fifth_column_1b[index] > 0 and ninety_fifth_column_1b[index] > 0:
            plot_draw = plt.bar(
                index, bottom=fifth_column_1b[index], height=ninety_fifth_column_1b[index]-fifth_column_1b[index], color='blue')
label_XXX = [i for i in range(len(mean_column))]

# Draw the mean dot on the graph
plt.plot(label_XXX,
         mean_column, "o", color="lime", label="Mean Point", markersize="4")

plt.title('Question1 (b)', color='black')
plt.legend()
plt.savefig("./Final_Exam/Q1_b.png")
plt.show()
####################################################################################################################################
####################################################################################################################################
# Question 1 c
####################################################################################################################################
####################################################################################################################################
num_X, size_p = X.shape
print(
    f"The length of the train_X is {num_X}, the size of each Train_X is {size_p}")
np.random.seed(20)
# This part of code is to the full training like b
coef_list_1c = list()
logisticRegression_1c_full_model = LogisticRegression(
    C=1000, solver='liblinear', penalty='l1')
logisticRegression_1c_full_model.fit(X, y.ravel())
# This is the parameter of DN
Jacknife_coef = logisticRegression_1c_full_model.coef_

# Here is to use jackknife method to solve
for index in range(num_X):
    # Using the other data to training like the previous hw cross-validation
    X_4_1c = np.delete(X, index, axis=0)
    y_4_1c = np.delete(y, index, axis=0)
    model_jackknife = LogisticRegression(
        C=1000, solver='liblinear', penalty='l1')
    model_jackknife.fit(X_4_1c, y_4_1c.ravel())
    # This is the result coef which is dn-1
    current_jackknife_coef = model_jackknife.coef_
    N_beta_DN = num_X * Jacknife_coef
    N_minus1_beta_DN_minus1 = (num_X-1) * current_jackknife_coef
    current_result = N_beta_DN - N_minus1_beta_DN_minus1
    coef_list_1c.append(current_result)

print(np.array(coef_list_1c).shape)
# Purify the structure of the list
purify_coeffcient_list_1c = list()
for item in coef_list_1c:
    for i in item:
        purify_coeffcient_list_1c.append(i)
print(np.array(purify_coeffcient_list_1c).shape)

# https://numpy.org/doc/stable/reference/generated/numpy.std.html
means_1c = np.mean(purify_coeffcient_list_1c, axis=0)
stdrs_1c = np.std(purify_coeffcient_list_1c, axis=0, ddof=1)
sqrt_square_stdrs = 1.645 * np.sqrt(np.square(stdrs_1c)/num_X)
fifth_column_1c = means_1c - sqrt_square_stdrs
ninety_fifth_column_1c = means_1c + sqrt_square_stdrs
# This is used to do the calculation for question 1 d
lower_1c = fifth_column_1c
upper_1c = ninety_fifth_column_1c

# Setting the color list for drawing the plot
color_list = list()
for index in range(size_p):
    if fifth_column_1c[index] <= 0 and ninety_fifth_column_1c[index] >= 0:
        color_list.append('red')
    else:
        color_list.append('blue')

# Generating the data list to draw the plot
bar_data_1c = list()
for index in range(len(fifth_column_1c)):
    bar_data_1c.append([fifth_column_1c[index], ninety_fifth_column_1c[index]])

# Checking the color of each bar
# using the absolute length to draw the bar plot
# Checking the span of each bar and print the color
for index in range(len(bar_data_1c)):
    ndarray_data = np.array(bar_data_1c[index])
    # This is the red bar
    # Check the bottom and the height of each bar
    if fifth_column_1c[index] <= 0 and ninety_fifth_column_1c[index] >= 0:
        plot_draw = plt.bar(
            index, bottom=fifth_column_1c[index], height=ninety_fifth_column_1c[index]+abs(fifth_column_1c[index]), color='red')
    else:
        # This is the blue bar
        if fifth_column_1c[index] < 0 and ninety_fifth_column_1c[index] < 0:
            plot_draw = plt.bar(
                index, bottom=ninety_fifth_column_1c[index], height=fifth_column_1c[index]-ninety_fifth_column_1c[index], color='blue')
        if fifth_column_1c[index] > 0 and ninety_fifth_column_1c[index] > 0:
            plot_draw = plt.bar(
                index, bottom=fifth_column_1c[index], height=ninety_fifth_column_1c[index]-fifth_column_1c[index], color='blue')
label_XXX = [i for i in range(len(means_1c))]

# Draw the mean dot on the graph
plt.plot(label_XXX,
         means_1c, "o", color="lime", label="Mean Point", markersize="4")

plt.title('Question1 (c)', color='black')
plt.legend()
plt.savefig("./Final_Exam/Q1_c.png")
plt.show()
####################################################################################################################################
####################################################################################################################################
# Question 1 d
####################################################################################################################################
####################################################################################################################################


def calculator_1d(lower_boundry, upper_boundry):
    # These are the parameter provided by the instruction
    np.random.seed(125)
    p = 30
    k = 8
    betas = np.random.random(p) + 1
    betas[np.random.choice(np.arange(p), p-k, replace=False)] = 0.0
    # Setting the confidence interval list to store the data
    CI_list = list()
    for index in range(p):
        # if the currend data is on the boundry setting the CI as 0
        if lower_boundry[index] <= 0 and upper_boundry[index] >= 0:
            CI_list.append(0)
        else:
            # Current the data not on the boundry setting the result as 1
            CI_list.append(1)
    # Change the current list to ndarray
    CI_array = np.array(CI_list)
    # Finding the location which CI_array value is 1 ----> in CI range
    index_eqls_1 = np.where(CI_array == 1)
    # This is to check the result is correct or not
    # print(betas[index_eqls_1] == 0)
    # print(np.where(betas[index_eqls_1] == 0)[0])
    # Finding the positive location
    False_Positive = np.where(betas[index_eqls_1] == 0)[0].shape
    # Finding the location which CI_array value is 0 ----> out of CI range
    index_eqls_0 = np.where(CI_array == 0)
    # Finding the negative location
    False_Negative = np.where(betas[index_eqls_0] != 0)[0].shape
    return False_Positive, False_Negative


FPa, FNa = calculator_1d(lower_1a, upper_1a)
FPb, FNb = calculator_1d(lower_1b, upper_1b)
FPc, FNc = calculator_1d(lower_1c, upper_1c)

print(f"FP of Question a {FPa[0]}, FN of Question a {FNa[0]}")
print(f"FP of Question a {FPb[0]}, FN of Question a {FNb[0]}")
print(f"FP of Question a {FPc[0]}, FN of Question a {FNc[0]}")
