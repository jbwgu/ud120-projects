import sys

#sys.path.append("../tools/")
sys.path.insert(0, 'C:/Users/WorkStation/Documents/GitHub/ud120-projects/tools')
# sys.path.insert(0, 'C:/Users/jtayl/Documents/GitHub/ud120-projects/tools')

# from class_vis import prettyPicture
# from prep_terrain_data import makeTerrainData

# import matplotlib.pyplot as plt
# import copy
# import numpy as np
# import pylab as pl


# features_train, labels_train, features_test, labels_test = makeTerrainData()


# ########################## SVM #################################
# ### we handle the import statement and SVC creation for you here
# from sklearn.svm import SVC
# clf = SVC(kernel="linear")


# #### now your job is to fit the classifier
# #### using the training features/labels, and to
# #### make a set of predictions on the test data



# #### store your predictions in a list named pred





# from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred, labels_test)

# def submitAccuracy():
#     return acc

# data = [115, 140, 175]

# scaled_data = []

# for item in data:
#     scaled_data.append((item - min(data)) / float(max(data) - min(data)))

# import nltk
# nltk.download()


from nltk.corpus import stopwords
sw = stopwords.words("english")
print(len(sw))
