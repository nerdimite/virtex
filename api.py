import os
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

def generate_data(mode='realtime', pose='Some Pose', save_to='./data/', save_as='null', nb_frames=1000):
    if mode == 'realtime':
        script = 'run_datagen_webcam'
        nb_frames = '--nb-frames {}'.format(nb_frames)
    elif mode == 'image':
        script = 'run_datagen_directory'
        nb_frames = ''

    cwd = os.path.join(os.getcwd(), "{}.py --pose {} --save-to {} --save-as {} {}".format(script, pose, save_to, save_as, nb_frames))
    os.system('{} {}'.format('python', cwd))

def preprocess_data(dataframe):
    # To Do return a dictionary mapping the classes with numbers
    # Importing the dataset
    x = dataframe.iloc[:, 0:-1].values
    y = dataframe['Pose'].values

    # Encoding classes
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # Get the mapping from encoded values to class name
    classes = {}
    for c in labelencoder_y.classes_:
        classes.update({labelencoder_y.transform([c])[0]:c})

    return x, y, classes

def train_classifier(x,y,save_as='classifier',save_to='./classifiers'):
    # Fitting XGBoost to the Training set
    classifier = XGBClassifier()
    classifier.fit(x, y)
    if os.path.isdir(save_to):
        pass
    else:
        os.makedirs(save_to)

    model_path = os.path.join(save_to,save_as)
    # Save classifier
    with open('{}.xgbc'.format(model_path), 'wb') as model:
        pickle.dump(classifier, model)

    return classifier

def cross_val(model,x,y):
    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator = model, X = x, y = y, cv = 10)
    return accuracies.mean(), accuracies.std()

def predict(mode='realtime', classifier='./classifiers/classifier.xgbc', cls_dict='./classifier/classifier.xgbc', save_to='./results/', save_as='null'):
    if mode == 'realtime':
        script = 'run_predict_webcam'
        save_as = ''
        save_to = ''
    elif mode == 'image':
        script = 'run_predict_directory'
        save_as = '--save-as {}'.format(save_as)
        save_to = '--save-to {}'.format(save_to)

    cwd = os.path.join(os.getcwd(), "{}.py --classifier {} --cls-dict {} {} {}".format(script, classifier, cls_dict, save_as, save_to))
    os.system('{} {}'.format('python', cwd))
