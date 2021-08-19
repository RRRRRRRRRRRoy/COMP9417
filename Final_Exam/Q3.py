import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = np.array(pd.read_csv("./Final_Exam/Data/Q3X.csv"))
y = np.array(pd.read_csv("./Final_Exam/Data/Q3y.csv"))
b = np.array([np.array([1]) for i in range(len(X))])

####################################################################################################################################
####################################################################################################################################
# Question 3 a
####################################################################################################################################
####################################################################################################################################

# This part of code is from the pseudo-code


def perceptron(X, y, max_iter=100):
    np.random.seed(1)
    w_0 = np.array([0, 0, 0])
    counter = 0
    for nmb_iter in range(1, max_iter):
        for index in range(len(X)):
            # W times X
            W_X = w_0 @ X[index]
            y_W_x = y[index] * W_X
            # This y structure is [0]
            # Getting the value in y
            current_value_y = y_W_x[-1]
            if current_value_y <= 0:
                y_X = y[index] * X[index]
                w_0 = w_0 + y_X
                counter += 1
    return w_0, counter


def plot_perceptron(ax, X, y, w):
    pos_points = X[np.where(y == 1)[0]]
    neg_points = X[np.where(y == -1)[0]]
    ax.scatter(pos_points[:, 1], pos_points[:, 2], color='blue')
    ax.scatter(neg_points[:, 1], neg_points[:, 2], color='red')
    xx = np.linspace(-6, 6)
    yy = -w[0]/w[2] - w[1]/w[2] * xx
    ax.plot(xx, yy, color='orange')

    ratio = (w[2]/w[1] + w[1]/w[2])
    xpt = (-1*w[0] / w[2]) * 1/ratio
    ypt = (-1*w[0] / w[1]) * 1/ratio

    ax.arrow(xpt, ypt, w[1], w[2], head_width=0.2, color='orange')
    ax.axis('equal')


w, nmb_iter = perceptron(X, y, max_iter=100)  # your trained perceptron
fig, ax = plt.subplots()
plot_perceptron(ax, X, y, w)  # from neural learning lab
ax.set_title(f"w={w}, iterations={nmb_iter}")
plt.savefig("name.png", dpi=300)  # if you want to save your plot as a png
plt.show()

####################################################################################################################################
####################################################################################################################################
# Question 3 b
####################################################################################################################################
####################################################################################################################################

alpha = np.array([np.array([0]) for i in range(len(X))])


####################################################################################################################################
####################################################################################################################################
# Question 3 c
####################################################################################################################################
####################################################################################################################################
