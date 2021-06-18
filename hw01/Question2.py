import numpy as np
import seaborn as sbn
import pandas as pd
import matplotlib.pyplot as plt

# Reading the data from the data.csv file
csv_data = pd.read_csv("./hw01/data.csv")
# print the result of reading
print(csv_data)

sbn.pairplot(csv_data)
plt.show()