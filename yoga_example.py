import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from classifier_utils import preprocess_data, train_classifier, cross_val
from utils import combine_poses, save_obj

'''
An Example Program to train on Yoga Poses
'''

dataset = combine_poses('.yoga_example/data/plank-0.csv','.yoga_example/data/tree-0.csv')
x_train, x_test, y_train, y_test, classes = preprocess_data(dataset)

classifier = train_classifier(x_train, y_train, 'yoga')

acc_mean, acc_std = cross_val(classifier,x_train,y_train)
print(acc_mean,acc_std)

save_obj(classes, './classifiers/yoga.cls')
