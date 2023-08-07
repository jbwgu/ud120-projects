#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot
# sys.path.append(os.path.abspath("../tools/"))
sys.path.insert(0, 'C:/Users/WorkStation/Documents/GitHub/ud120-projects/tools')
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load(open("C:/Users/WorkStation/Documents/GitHub/ud120-projects/final_project/final_project_dataset.pkl", "r"))
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

# Plot the data
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

# Find the key for the outlier
for key in data_dict:
    if data_dict[key]["salary"] != 'NaN' and data_dict[key]["salary"] > 10000000 and data_dict[key]["bonus"] != 'NaN' and data_dict[key]["bonus"] > 5000000:
        print(key)

# Remove the outlier
data_dict.pop("TOTAL", 0)

# Replot the data
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

# Two people made bonuses of at least 5 million dollars, and a salary of over 1 million dollars. What are the names associated with those points?

for key in data_dict:
    if data_dict[key]["salary"] != 'NaN' and data_dict[key]["salary"] > 1000000 and data_dict[key]["bonus"] != 'NaN' and data_dict[key]["bonus"] > 5000000:
        print(key)

