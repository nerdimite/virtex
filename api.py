import os
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

def generate_data(mode='realtime', pose='Some Pose', folder='./images/', save_to='./data/', save_as='null', nb_frames=1000, model='mobilenet_thin'):
    '''
    Saves the data of a single pose as a separate csv file which can be later used for preprocessing

    PARAMETERS:
    mode :- Key Point data can be generated in two ways, 'realtime' mode and 'image' model
                realtime - captures poses in realtime through a webcam
                image - captures poses from images in a folder
    pose :- name of the pose you are generating data for
    folder :- path to the folder of image (only for image mode)
    save_to :- path to the folder to save your csv files
    save_as :- name of the csv file you want to save it with
    nb_frames :- number of frames or entries you want to capture in realtime mode (only for realtime mode)
    model :- (only for realtime mode) choose from mobilenet_thin / mobilenet_v2_small / mobilenet_v2_larger / cmu. Default is mobilenet_thin
    '''
    if mode == 'realtime':
        script = 'run_datagen_webcam'
        nb_frames = '--nb-frames {}'.format(nb_frames)
        folder = ''
    elif mode == 'image':
        script = 'run_datagen_directory'
        nb_frames = ''
        folder = '--folder {}'.format(folder)
    else:
        raise ValueError('Please select one of two values for the parameter mode i.e. realtime / image')

    cwd = os.path.join(os.getcwd(), "{}.py --model {} {} --pose {} --save-to {} --save-as {} {}".format(script, model, folder, pose, save_to, save_as, nb_frames))
    os.system('{} {}'.format('python', cwd))

def preprocess_data(dataframe):
    '''
    RETURNS two arrays, x and y from the dataset and a dictionary, classes which is a mapping
            between the encoded integers and the pose name strings

    PARAMETERS:
    dataframe :- the dataframe containing the keypoint data of all the poses
    '''

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
    '''
    RETURNS classifier model

    PARAMETERS:
    x :- the input array of features
    y :- the target array of pose labels
    save_as :- the name of the model with which you want to save it
    save_to :- directory where you want to save the model to
    '''
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
    '''
    RETURNS two float values of mean accuracy and standard deviation

    PARAMETERS:
    model :- the trained classifier model
    x :- the input array of features
    y :- the target array of pose labels
    '''
    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator = model, X = x, y = y, cv = 10)
    return accuracies.mean(), accuracies.std()

def predict(mode='realtime', classifier='./classifiers/classifier.xgbc', cls_dict='./classifier/classifier.cls', save_to='./results/', save_as='null', model='mobilenet_thin'):
    '''
    Predicts the poses in realtime or on a batch of images

    PARAMETERS:
    mode :- poses can be predicted in two ways, 'realtime' mode and 'image' model
                realtime - infers poses in realtime through a webcam
                image - infers poses from images in a folder
    classifier :- path to the saved model
    cls_dict :- path to the dictionary object containing the mapping
    save_to :- path to the folder to save your results (only for image mode)
    save_as :- name of the csv file you want to save the results with (only for image mode)
    model :- choose from mobilenet_thin / mobilenet_v2_small / mobilenet_v2_larger / cmu. Default is mobilenet_thin
    '''
    if mode == 'realtime':
        script = 'run_predict_webcam'
        save_as = ''
        save_to = ''
    elif mode == 'image':
        script = 'run_predict_directory'
        save_as = '--save-as {}'.format(save_as)
        save_to = '--save-to {}'.format(save_to)
    else:
        raise ValueError('Please select one of two values for the parameter mode i.e. realtime / image')

    cwd = os.path.join(os.getcwd(), "{}.py --model {} --classifier {} --cls-dict {} {} {}".format(script, model, classifier, cls_dict, save_as, save_to))
    os.system('{} {}'.format('python', cwd))
