# COMP9417 Exam Template Code
# All code in PDF is copied here for ease of use. You can use multiple .py files in your submissions.

# Question 1
import pandas as pd  # not needed
from sklearn.linear_model import LinearRegression
import pandas as pd  # not really needed, only for preference
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
np.random.seed(125)
p = 20
k = 8
betas = np.random.random(p) + 1
betas[np.random.choice(np.arange(p), p-k, replace=False)] = 0.0

# Question 2 part (d)
matplotlib.rc('font', **{'size': 14})   # make plot text more visible
# do not import anything else


def mse(y, yhat):
    return np.mean(np.square(y-yhat))


def your_function(y):
    # your code here


    # load in data
y_dict = np.load('Q2_mono_dict.npy', allow_pickle=True).item()

# create plot
fig, ax = plt.subplots(3, 2, figsize=(14, 14))
plot_titles = ['(i)', '(ii)', '(iii)', '(iv)', '(v)', '(vi)']

for i, ax in enumerate(ax.flat):

    y = y_dict[plot_titles[i]]
    x = np.arange(y.shape[0])
    betahat = your_function(y)  # update this line

    # plot data and fit
    ax.scatter(x, y, marker='o', label="y")
    ax.plot(x, betahat,
            color='orange', linestyle='-', marker='s',
            label=r'$\hat{\beta}$')  # if you have latex issues, replace with label='hatbeta'

    # set title and put legend in a good spot
    mse_bh = np.round(mse(y, betahat), 4)
    ax.set_title(f'part={plot_titles[i]}, mse={mse_bh}')
    ax.legend(loc='best')

plt.tight_layout()
plt.savefig("pickAName.png", dpi=400)  # update this line
plt.show()


# Question 3


def perceptron(X, y, max_iter=100):
    np.random.seed(1)
    # your code here

    return w, nmb_iter


w =  # your trained perceptron
fig, ax = plt.subplots()
plot_perceptron(ax, X, y, w)       # from neural learning lab
ax.set_title(f"w={w},    iterations={nmb_iter}")
plt.savefig("name.png", dpi=300)      # if you want to save your plot as a png
plt.show()


# Question 4


def total_loss(X, y, Z, models):
    '''
    computes total loss achieved on X, y based on linear regressions stored in models, and partitioning Z

    :param X: design matrix, n x p (np.array shape (n,p))
    :param y: response vector, n x 1 (np.array shape (n,1) or (n,))
    :param Z: assignment vector, n x 1 (assigns each sample to a partition)
    :param models: a list of M sklearn LinearRegression models, one for each of the partitions

    :returns: the loss of your complete model as computed in (a)
    '''

    loss = 0
    M = len(models)

    # Your code here

    return loss


def find_partitions(X, y, models):
    '''
    given M models, assigns points in X to one of the M partitions

    :param X: design matrix, n x p (np.array shape (n,p))
    :param y: response vector, n x 1 (np.array shape (n,1) or (n,))
    :param models: a list of M sklearn LinearRegression models for each 
    of the partitions

    :returns: Z, a np.array of shape (n,) assigning each of the points in X to one of M partitions
    '''
    M = len(models)
    # your code here
    return Z
