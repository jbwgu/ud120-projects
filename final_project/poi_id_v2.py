import sys
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, 'C:/Users/WorkStation/Documents/GitHub/ud120-projects/tools')

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Set working directory
new_dir = "C:/Users/WorkStation/Documents/GitHub/ud120-projects/final_project/"
os.chdir(new_dir)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open('final_project_dataset.pkl', 'r') as data_file:
    data_dict = pickle.load(data_file)

my_dataset2 = data_dict

### Task 2: Remove outliers
for key in data_dict:
    for feature in features_list:
        # Skip the poi label
        if feature == 'poi':
            continue
        # elif feature == 'salary':
        # Get the values for the feature
        values = []
        for key in data_dict:
            if data_dict[key][feature] != 'NaN':
                values.append(data_dict[key][feature])
        # Calculate the Tukey fences
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        upper_fence = Q3 + 1.5 * IQR
        lower_fence = Q1 - 1.5 * IQR
        # Remove the outliers
        for key in data_dict:
            if data_dict[key][feature] != 'NaN':
                if data_dict[key][feature] > upper_fence or data_dict[key][feature] < lower_fence:
                    data_dict[key][feature] = 'NaN'

### Task 3: Create new feature(s)
# Fraction of emails this person received that were from POIs
for name in data_dict.keys():
    if data_dict[name]['from_poi_to_this_person'] == 'NaN' or data_dict[name]['to_messages'] == 'NaN':
        data_dict[name]['fraction_from_poi'] = 'NaN'
    else:
        fraction = float(data_dict[name]['from_poi_to_this_person']) / float(data_dict[name]['to_messages'])
        data_dict[name]['fraction_from_poi'] = fraction

# New feature 2: fraction_to_poi
# Fraction of emails this person sent that were to POIs
for name in data_dict.keys():
    if data_dict[name]['from_this_person_to_poi'] == 'NaN' or data_dict[name]['from_messages'] == 'NaN':
        data_dict[name]['fraction_to_poi'] = 'NaN'
    else:
        fraction = float(data_dict[name]['from_this_person_to_poi']) / float(data_dict[name]['from_messages'])
        data_dict[name]['fraction_to_poi'] = fraction

# New feature 3: total_compensation
# Total compensation of each employee, including salary, bonus, and long-term incentive
for name in data_dict.keys():
    total = 0
    if data_dict[name]['salary'] != 'NaN':
        total += data_dict[name]['salary']
    if data_dict[name]['bonus'] != 'NaN':
        total += data_dict[name]['bonus']
    if data_dict[name]['long_term_incentive'] != 'NaN':
        total += data_dict[name]['long_term_incentive']
    data_dict[name]['total_compensation'] = total

# # New feature 4: deferred_compensation_ratio
# # Ratio of deferred compensation to total compensation
# for name in data_dict.keys():
#     if data_dict[name]['total_compensation'] == 'NaN' or data_dict[name]['deferred_income'] == 'NaN' or data_dict[name]['deferral_payments'] == 'NaN':
#         data_dict[name]['deferred_compensation_ratio'] = 'NaN'
#     else:
#         total = float(data_dict[name]['total_compensation'])
#         deferred = float(data_dict[name]['deferred_income']) + float(data_dict[name]['deferral_payments'])
#         ratio = deferred / total
#         data_dict[name]['deferred_compensation_ratio'] = ratio

# New feature 5: email_interaction_ratio
# Ratio of emails sent to POIs to emails received from POIs
for name in data_dict.keys():
    if data_dict[name]['from_poi_to_this_person'] == 'NaN' or data_dict[name]['from_this_person_to_poi'] == 'NaN' or data_dict[name]['to_messages'] == 'NaN' or data_dict[name]['from_messages'] == 'NaN':
        data_dict[name]['email_interaction_ratio'] = 'NaN'
    else:
        sent_to_poi = float(data_dict[name]['from_this_person_to_poi'])
        received_from_poi = float(data_dict[name]['from_poi_to_this_person'])
        sent_total = float(data_dict[name]['from_messages'])
        received_total = float(data_dict[name]['to_messages'])
        if received_from_poi == 0:
            ratio = 0
        else:
            ratio = (sent_to_poi + received_from_poi) / (sent_total + received_total)
        data_dict[name]['email_interaction_ratio'] = ratio

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Split the data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# Create the pipeline with SVM classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', PCA()),
    ('selector', SelectKBest(k='all')),
    ('classifier', SVC())
])

# Set the parameters to tune
parameters = {
    'reduce_dim__n_components': [2, 4, 6],
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': [0.1, 0.01, 0.001]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, parameters)

# Fit the GridSearchCV object to the training data
grid_search.fit(features_train, labels_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Use the best estimator to make predictions on the test data
predictions = best_estimator.predict(features_test)

# Calculate precision and recall
precision = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)

print('Best Parameters:', best_params)
print('Precision:', precision)
print('Recall:', recall)

# Dump the best classifier and dataset
dump_classifier_and_data(best_estimator, my_dataset, features_list)
