# Importing the libraries
import os
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

def preprocess_data(dataframe, test_split=0.2):
    # To Do return a dictionary mapping the classes with numbers
    # Importing the dataset
    X = dataframe.iloc[:, 0:-1].values
    y = dataframe['Pose'].values

    # Encoding classes
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # Get the mapping from encoded values to class name
    classes = {}
    for c in labelencoder_y.classes_:
        classes.update({labelencoder_y.transform([c])[0]:c})

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_split, random_state = 0)

    return X_train, X_test, y_train, y_test, classes

def train_classifier(x,y,model_name='classifier'):
    # Fitting XGBoost to the Training set
    classifier = XGBClassifier()
    classifier.fit(x, y)
    if os.path.isdir('classifiers'):
        pass
    else:
        os.makedirs('classifiers')
    # Save classifier
    with open('./classifiers/{}.model'.format(model_name), 'wb') as model:
        pickle.dump(classifier, model)
    
    return classifier

def cross_val(model,x,y):
    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator = model, X = x, y = y, cv = 10)
    return accuracies.mean(), accuracies.std()
