'''
Example Usage of the Virtex API to generate data, train, and predict poses
'''

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from time import sleep
from api import preprocess_data, train_classifier, cross_val, generate_data, predict
from utils import combine_poses, save_obj

# Data can be generated in two modes. Refer the documentation for more info on parameters and modes.
# In this example we are going to use the realtime mode
print('Capturing standing pose in 3 seconds')
sleep(3)
generate_data('realtime', pose='standing', save_to='./example/data/', save_as='standing.csv', nb_frames=100)
print('Capturing sitting pose in 3 seconds')
sleep(3)
generate_data('realtime', pose='sitting', save_to='./example/data/', save_as='sitting.csv', nb_frames=100)

# Now we combine the different csv files of generated for each pose into one dataframe
dataset = combine_poses('./example/data/standing.csv','./example/data/sitting.csv')
# You can also save the combined dataframe to csv if you want to, for future use
# dataset.to_csv('./example/data/pose.csv', index=False)

# We can use the preprocess_data function to encode the poses and get the x, y and classes dictionary
# containing the mapping from encoded numbers to pose names
x_train, y_train, classes = preprocess_data(dataset)

# Use the train_classifier function to begin training the model
classifier = train_classifier(x_train, y_train, save_as='pose', save_to='./example/classifiers')

# k-fold cross validation is available to get the accuracy of the trained model
acc_mean, acc_std = cross_val(classifier,x_train,y_train)
print(acc_mean,acc_std)

# Save the classes dictionary to a file so that we can retrieve it later during prediction
save_obj(classes, './example/classifiers/pose.cls')

# To make the predictions, there are two modes as well (refer documentation)
# For now we will do realtime prediction of poses
predict(mode='realtime',classifier='./example/classifiers/pose.xgbc', cls_dict='./example/classifiers/pose.cls')
