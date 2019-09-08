import argparse
import logging
import time

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from classifier_utils import preprocess_data, train_classifier, cross_val
from utils import combine_poses, save_obj



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='virtex training classifier')
    parser.add_argument('--dataset', required=True, type=str, help='the path to the csv file dataset of all poses')
    parser.add_argument('--save-model-as', type=str, help='the name with which you want to save to classifier')
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    x, y, classes = preprocess_data(dataset)
    classifier = train_classifier(x, y, 'args.save_model_as')

    acc_mean, acc_std = cross_val(classifier,x,y)
    print('k-fold Cross Validation Results')
    print('Mean Accuracy: {}'.format(acc_mean))
    print('Standard Deviation of Accuracy: {}'.format(acc_std))

    save_obj(classes, './classifiers/{args.save_model_as}.cls')
