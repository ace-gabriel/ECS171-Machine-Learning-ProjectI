"""
   1. mpg:           continuous
   2. cylinders:     multi-valued discrete
   3. displacement:  continuous
   4. horsepower:    continuous
   5. weight:        continuous
   6. acceleration:  continuous
   7. model year:    multi-valued discrete
   8. origin:        multi-valued discrete
   9. car name:      string (unique for each instance)

"""
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Problem 1

#file = open("/Users/gabriel/Desktop/auto-mpg.data.txt")
#db = file.readlines()
# Initialize lists
#mpg, cylin, disp, horsep, weig, accele, year, origin, name, mpg_buff = [],[],[],[],[],[],[],[],[],[]

# read file
file_ = pd.read_csv("auto-mpg.data.txt", sep='\s+', header = None)
feature_map = ["mpg", "cylin", "disp", "horsep", "weight", "accele", "year", "origin", "name"]
file_.columns = feature_map # map file to feature map
file_ = file_.replace({'?': np.nan}).dropna() # replace and delete garbage value

# sort the data and put determine the threshold
table = file_.sort_values("mpg", axis = 0, ascending = True, kind = 'heapsort')
table.reset_index(drop=True, inplace=True)

# returns threshold
low = float(table.get_value(130, "mpg", takeable = False)) # 18.6
medium = float(table.get_value(260, "mpg", takeable = False)) # 26.8

# Problem 2

"""for i in range(0,392): # make the chart

    sub.append(mpg[i])
    sub.append(cylin[i])
    sub.append(disp[i])
    sub.append(horsep[i])
    sub.append(weig[i])
    sub.append(accele[i])
    sub.append(year[i])
    sub.append(origin[i])
    sub.append(target[i])
    result.append(sub)
    sub = [] # reset


features = ['mpg','cylin','disp','horsep','weig','accele','year','origin','identifier']
data = np.asarray(result) """

# convert horsepower value into float
file_['horsep'] = pd.to_numeric(file_['horsep'].str.replace(' ', ''), errors='force')
table['horsep'] = pd.to_numeric(table['horsep'].str.replace(' ', ''), errors='force')

# color wheel to determine the output
color_wheel = {1: "red",
               2: "green",
               3: "blue"}
"""
fm = pd.DataFrame(data, columns = features)
# Add an identifier to render data points
colors = fm['identifier'].map(lambda x: color_wheel.get(x + 1))
scatter_matrix(fm, color = colors, figsize = (10,5), alpha = 0.5, diagonal = 'hist') """


# assign color based on different categories
table['color'] = pd.Series("green", index = table.index)
for i in range(0, 392):
    if i > 130 and i <= 260:
        table.at[i, "color"] = "blue"
    elif i > 260:
        table.at[i, "color"] = "red"

# set up a scatter matrix and plot the scatter matrix
scatter_matrix(table, c = table[table.columns[-1]], figsize = (10,5), alpha = 0.5, diagonal = 'hist')
plt.show()

# Please see problem3-5.py and problem6-7.py
