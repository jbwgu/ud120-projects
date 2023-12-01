import sys
import pickle
import os
import numpy as np
import warnings
# Suppress the FutureWarning for n_estimators
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

# Modify path for review
# sys.path.insert(0, 'C:/Users/WorkStation/Documents/GitHub/ud120-projects/tools')
sys.path.append(os.path.abspath(("../tools/")))
print(sys.path)
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Set working directory
# Modified for review
# new_dir = "C:/Users/WorkStation/Documents/GitHub/ud120-projects/final_project/"
# os.chdir(new_dir)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'long_term_incentive', 
                 'restricted_stock', 'director_fees', 'to_messages', 
                 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

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
                if data_dict[key][feature] > upper_fence \
                    or data_dict[key][feature] < lower_fence:
                    data_dict[key][feature] = 'NaN'

### Task 3: Create new feature(s)
# Fraction of emails this person received that were from POIs
for name in data_dict.keys():
    if data_dict[name]['from_poi_to_this_person'] == 'NaN' \
    or data_dict[name]['to_messages'] == 'NaN':
        data_dict[name]['fraction_from_poi'] = 'NaN'
    else:
        fraction = float(data_dict[name]['from_poi_to_this_person']) \
            / float(data_dict[name]['to_messages'])
        data_dict[name]['fraction_from_poi'] = fraction

# New feature 2: fraction_to_poi
# Fraction of emails this person sent that were to POIs
for name in data_dict.keys():
    if data_dict[name]['from_this_person_to_poi'] == 'NaN' \
        or data_dict[name]['from_messages'] == 'NaN':
        data_dict[name]['fraction_to_poi'] = 'NaN'
    else:
        fraction = float(data_dict[name]['from_this_person_to_poi']) \
            / float(data_dict[name]['from_messages'])
        data_dict[name]['fraction_to_poi'] = fraction

# New feature 3: total_compensation
# Total compensation of each employee, including salary, bonus, and long-term 
# incentive
for name in data_dict.keys():
    total = 0
    if data_dict[name]['salary'] != 'NaN':
        total += data_dict[name]['salary']
    if data_dict[name]['bonus'] != 'NaN':
        total += data_dict[name]['bonus']
    if data_dict[name]['long_term_incentive'] != 'NaN':
        total += data_dict[name]['long_term_incentive']
    data_dict[name]['total_compensation'] = total

# New feature 4: email_interaction_ratio
# Ratio of emails sent to POIs to emails received from POIs
for name in data_dict.keys():
    if data_dict[name]['from_poi_to_this_person'] == 'NaN' \
        or data_dict[name]['from_this_person_to_poi'] == 'NaN' \
            or data_dict[name]['to_messages'] == 'NaN' \
                or data_dict[name]['from_messages'] == 'NaN':
        data_dict[name]['email_interaction_ratio'] = 'NaN'
    else:
        sent_to_poi = float(data_dict[name]['from_this_person_to_poi'])
        received_from_poi = float(data_dict[name]['from_poi_to_this_person'])
        sent_total = float(data_dict[name]['from_messages'])
        received_total = float(data_dict[name]['to_messages'])
        if received_from_poi == 0:
            ratio = 0
        else:
            ratio = (sent_to_poi + received_from_poi) \
                / (sent_total + received_total)
        data_dict[name]['email_interaction_ratio'] = ratio

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# Split the data into training and testing sets
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a pipeline with feature scaling and classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', None)
])

# Define a list of classifiers to try
classifiers = [
    ('Support Vector Machine', SVC()),
    ('Random Forest', RandomForestClassifier()),
    ('Logistic Regression', LogisticRegression(solver='lbfgs')),
    ('Gaussian Naive Bayes', GaussianNB())
]

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Iterate over the classifiers and fit the pipeline
for name, classifier in classifiers:
    print('Training', name)
    if name == 'Support Vector Machine':
        pipeline.set_params(classifier=svm.SVC())
    elif name == 'Random Forest':
        pipeline.set_params(classifier=RandomForestClassifier(n_estimators=100))
    elif name == 'Logistic Regression':
        pipeline.set_params(classifier=LogisticRegression(solver='lbfgs'))
    else:
        pipeline.set_params(classifier=classifier)
    pipeline.set_params(classifier=classifier)
    pipeline.fit(features_train, labels_train)
    accuracy = pipeline.score(features_test, labels_test)
    print('Accuracy:', accuracy)
    predictions = pipeline.predict(features_test)
    if 1 in predictions:
        precision = precision_score(labels_test, predictions)
    else:
        precision = 0.0
    print('Precision:', precision)
     # Calculate recall
    recall = recall_score(labels_test, predictions)
    print('Recall:', recall)
    if name == 'Random Forest':
        final_model = pipeline.named_steps['classifier']
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print('Feature Ranking:')
        for i in range(18):
            print('{} feature {} ({})'.format(i+1, features_list[i+1], \
                importances[indices[i]]))

        dump_classifier_and_data(pipeline, my_dataset, features_list)