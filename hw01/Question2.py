import numpy as np
import seaborn as sbn
import pandas as pd

# Reading the data from the data.csv file
csv_data = pd.read_csv("./data.csv")
# print the result of reading
print(csv_data)

sbn.pairplot(csv_data)