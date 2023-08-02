#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
import sys
from time import time
#sys.path.append("../tools/")
sys.path.insert(0, 'C:/Users/WorkStation/Documents/GitHub/ud120-projects/tools')
# sys.path.insert(0, 'C:/Users/jtayl/Documents/GitHub/ud120-projects/tools')

enron_data = joblib.load(open("C:/Users/WorkStation/Documents/GitHub/ud120-projects/final_project/final_project_dataset.pkl", "r"))

# how many data points (people) are in the dataset?
print("Number of people: ", len(enron_data))

# for each person, how many features are available?
print("Number of features: ", len(enron_data["SKILLING JEFFREY K"]))

# how many POIs are there in the E+F dataset?
poi_count = 0
for key in enron_data:
    if enron_data[key]["poi"] == 1:
        poi_count += 1
print("Number of POIs: ", poi_count)

# how many POIs are there in total?
poi_names = open("C:/Users/WorkStation/Documents/GitHub/ud120-projects/final_project/poi_names.txt", "r")
poi_names = poi_names.readlines()
poi_count = 0

for line in poi_names:
    poi_count += 1
print("Number of POIs: ", poi_count)

print("Total Stock Value for James Prentice: ", enron_data["PRENTICE JAMES"]["total_stock_value"])

print("Number of emails from Wesley Colwell to POIs: ", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

print("Value of stock options exercised by Jeffrey K Skilling: ", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

print("Total Payments for Jeffrey K Skilling: ", enron_data["SKILLING JEFFREY K"]["total_payments"])
print("Total Payments for Kenneth Lay: ", enron_data["LAY KENNETH L"]["total_payments"])
print("Total Payments for Andrew Fastow: ", enron_data["FASTOW ANDREW S"]["total_payments"])

# how many folks in this dataset have a quantified salary? What about a known email address?
salary_count = 0
email_count = 0
for key in enron_data:
    if enron_data[key]["salary"] != "NaN":
        salary_count += 1
    if enron_data[key]["email_address"] != "NaN":
        email_count += 1
print("Number of people with quantified salaries: ", salary_count)
print("Number of people with known email addresses: ", email_count)

# how many people in the E+F dataset (as it currently exists) have "NaN" for their total payments?
# what percentage of people in the dataset as a whole is this?
total_payments_count = 0
for key in enron_data:
    if enron_data[key]["total_payments"] == "NaN":
        total_payments_count += 1
print("Number of people with NaN for their total payments: ", total_payments_count)
print("Percentage of people with NaN for their total payments: ", float(total_payments_count)/len(enron_data)*100)

# how many POIs in the E+F dataset have "NaN" for their total payments?
# what percentage of POI's as a whole is this?
poi_total_payments_count = 0
for key in enron_data:
    if enron_data[key]["poi"] == 1:
        if enron_data[key]["total_payments"] == "NaN":
            poi_total_payments_count += 1
print("Number of POIs with NaN for their total payments: ", poi_total_payments_count)

# if 10 POI's were added to the dataset, with "NaN" for total payments, what new percentage of POI's have "NaN" for total payments?
print("New percentage of POIs with NaN for their total payments: ", float(poi_total_payments_count+10)/len(enron_data)*100)

# what is the new number of people with "NaN" for their total payments?
print("New number of people with NaN for their total payments: ", total_payments_count+10)

# what is the new number of POI's with "NaN" for their total payments?
print("New number of POIs with NaN for their total payments: ", poi_total_payments_count+10)
