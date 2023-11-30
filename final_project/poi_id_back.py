#!/usr/bin/python

import sys
import pickle
import os
import numpy as np

sys.path.insert(0, 'C:/Users/WorkStation/Documents/GitHub/ud120-projects/tools')

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Set working directory
new_dir = "C:/Users/WorkStation/Documents/GitHub/ud120-projects/final_project/"
os.chdir(new_dir)

### Task 1: Select what features you'll use.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
                 'expenses', 'exercised_stock_options', 'long_term_incentive', 
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open('final_project_dataset.pkl', 'r') as data_file:
    #data_dict = pickle.load(io.BytesIO(data_file.read()))
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

# New feature 1: fraction_from_poi
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
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# # Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm

# Split the data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a pipeline with feature scaling and classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', None)
])

# Define a list of classifiers to try
classifiers = [
    ('Support Vector Machine', SVC()),
    ('Random Forest', RandomForestClassifier()),
    ('Logistic Regression', LogisticRegression())
    # ,('Gaussian Naive Bayes', GaussianNB())
]

# Iterate over the classifiers and fit the pipeline
for name, classifier in classifiers:
    # print('Training', name)
    if name == 'Support Vector Machine':
        pipeline.set_params(classifier=svm.SVC())
    elif name == 'Random Forest':
        pipeline.set_params(classifier=RandomForestClassifier(n_estimators=100))
    elif name == 'Logistic Regression':
        pipeline.set_params(classifier=LogisticRegression(solver='lbfgs'))
    else:
        pipeline.set_params(classifier=classifier)
    # pipeline.set_params(classifier=classifier)
    pipeline.fit(features_train, labels_train)
    accuracy = pipeline.score(features_test, labels_test)
    # print('Accuracy:', accuracy)
    predictions = pipeline.predict(features_test)
    if 1 in predictions:
        #precision = precision_score(labels_test, predictions, zero_division=0)
        precision = precision_score(labels_test, predictions)
    else:
        precision = 0.0
    
    # print('Precision:', precision)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV

# Set the parameters to tune
parameters = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': [0.1, 1, 10]
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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
