# Virtex 1.0
## End-to-End Pose Recognition

Virtex is an easy to use API for pose recognition, built on top of this awesome person's [tf_pose_estimation](https://github.com/ildoonet/tf-pose-estimation), which lets you estimate and classify the poses.
You can train on a bulk of images or you can just train for the poses in real-time using the webcam data generator. The classifier implements XG Boost Classifier to train and make predictions.

[Medium Post of Virtex](https://medium.com/@bhavesh.laddagiri1/how-to-implement-pose-recognition-in-10-lines-of-code-using-virtex-9bd26cf451db)

## Table of Contents

* [Installation](#install)
* [Tutorial](#tut)
* [Documentation (API)](#apidocs)
    * [Data Generation](#gen)
    * [Preprocessing](#pre)
    * [Training](#train)
    * [Inference](#pred)
* [Documentation (CLI)](#clidocs)
* [Future Updates](#feat)
* [Contribute](#ctrbte)
* [References](#ref)

<a name="install"/>

### Installation

#### Install

You need following dependencies

- python 3
- tensorflow 1.4.1 and above
- opencv3, protobuf, python3-tk
- slidingwindow
- xgboost

Clone the repo and install 3rd-party libraries.

```bash
$ git clone https://github.com/theneuralbeing/virtex.git
$ cd virtex
$ pip3 install -r requirements.txt
```
Build c++ library for post processing. See : https://github.com/theneuralbeing/virtex/tree/master/tf_pose/pafprocess
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```
Alternatively, you can install this repo as a shared package using pip.

```bash
$ git clone https://github.com/theneuralbeing/virtex.git
$ cd virtex
$ python setup.py install  # Or, `pip install -e .`
```
#### Download Tensorflow Graph File(pb file)

The available models are

- cmu (trained in 656x368)
- mobilenet_thin (trained in 432x368)
- mobilenet_v2_large (trained in 432x368)
- mobilenet_v2_small (trained in 432x368)

CMU's model graphs are too large for git, so they are uploaded on an external cloud. You should download them if you want to use cmu's original model. Download scripts are provided in the model folder.

```
$ cd models/graph/cmu
$ bash download.sh
```
<a name="tut"/>

### Tutorial

I have written a [blog post on medium](https://medium.com/@bhavesh.laddagiri1/how-to-implement-pose-recognition-in-10-lines-of-code-using-virtex-9bd26cf451db) detailing on the usage of Virtex for realtime pose recognition. Make sure you check it out. Also, the Jupyter Notebook associated with the blog can be found [here](https://github.com/theneuralbeing/virtex/blob/master/Tutorial.ipynb)

<a name="apidocs"/>

### Documentation (API)

Virtex comes with an easy to use api which does all the heavy lifting of generating, processing, training and inferencing poses. Pose Recognition with virtex is done in four simple steps:

- Generation
- PreProcessing
- Training
- Inference

<a name="gen"/>

#### Data Generation

Before you start recognising poses, the keypoint data of the poses needs to be processed into a csv file.
The `generate_data()` function returns the individual pose's csv files ready to be processed
```python
Usage:

from api import generate_data
generate_data(PARAMETERS)

Docstring:
    Saves the data of a single pose as a separate csv file which can be later used for preprocessing

    PARAMETERS:
    mode :- Key Point data can be generated in two ways, 'realtime' mode and 'image' model
                realtime - captures poses in realtime through a webcam
                image - captures poses from images in a folder
    pose :- name of the pose you are generating data for
    folder :- path to the folder if you are using image model
    save_to :- path to the folder to save your csv files
    save_as :- name of the csv file you want to save it with
    nb_frames :- number of frames or entries you want to capture in realtime mode
    model :- only for realtime mode, accepts mobilenet_thin / mobilenet_v2_small / mobilenet_v2_larger / cmu. Default is mobilenet_thin
```
<a name="pre"/>

#### Preprocessing

The `generate_data()` function creates a separate csv files for each pose. To get the final dataset, you need to combine
the csv files into a single dataframe. This can be achieved using the `combine_poses(*args)` function from `utils`.
```python
Usage:

from utils import combine_poses
dataframe = combine_poses(PARAMETERS)

Docstring:
    RETURNS the dataframe containing all the poses' data

    PARAMETERS:
    *args :- The file paths of all the pose csv files.
    Example Usage:
    data = combine_poses('./data/pose-1.csv','./data/pose-2.csv',.....,'./data/pose-n.csv')
    You can add as many paths to the csv files
    
```
The dataframe returned by `combine_poses()` is now ready to be fed for preprocessing using the `preprocess_data(dataframe)`
function which splits the data as an array of x features and y labels, encodes the labels and returns a dictionary containing
the mapping from encoded label to string label (eg: {0:'pose-1', 1: 'pose-2'}).

```python
Usage:

from api import preprocess_data
x_train, y_train, classes = preprocess_data(PARAMETERS)

Docstring:
    RETURNS two arrays, x and y from the dataset and a dictionary, classes which is a mapping
            between the encoded integers and the pose name strings

    PARAMETERS:
    dataframe :- the dataframe containing the keypoint data of all the poses
```
Additionally you can save the dataframe to csv as usual using pandas (eg: `dataframe.to_csv('data.csv', index=False)`)

<a name="train"/>

#### Training

The training process is as simple as a single line of code. Currently, the API has XGBoost Classifier with default hyperparameters, but more models like SVMs, Bayesian Models and Neural Nets will be added in the future. The `train_classifier()` function returns the trained model and also saves it by default.

```python
Usage:

from api import train_classifier
classifier = train_classifier(PARAMETERS)

Docstring:
    RETURNS trained model

    PARAMETERS:
    x :- the input array of features
    y :- the target array of pose labels
    save_as :- the name of the model with which you want to save it
    save_to :- directory where you want to save the model to
```
The accuracy of the trained model can be obtained by the `cross_val()` function which implements k-fold cross validation.

```python
Usage:
from api import cross_val
acc_mean, acc_std = cross_val(PARAMETERS)

Docstring:
    RETURNS two float values of mean accuracy and standard deviation

    PARAMETERS:
    model :- the trained classifier model
    x :- the input array of features
    y :- the target array of pose labels
```

<a name="pred"/>

#### Inference

The `predict()` function makes it super easy to make realtime inference as well as inference on a batch of images. If you are
inferencing on a batch of images, the results are saved for you in a csv file containing the image name and the predicted pose.
```python
Usage:
from api import predict
predict(PARAMETERS)

Docstring:
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
```

<a name="clidocs"/>

### Documentation (CLI)

The API runs the CLI based scripts behind the scenes so you can infact directly generate, train and predict using the CLI
based scripts. The following scripts are available with CLI:

- run_datagen_directory.py
- run_datagen_webcam.py
- run_train_classifier.py
- run_predict_directory.py
- run_predict_webcam.py

### Future Updates

- More Classifiers (SVM, DNN)
- Inference Logging and continuous training to decrease prediction fluctuations during inference and increase stability and overall performance
- Support for Continuous Actions using LSTMs
- Hand Gesture Recognition
- Tracking specific Tagged Persons using Face Recognition
- 3D Pose

<a name="ctrbte"/>

### Contribute

Feel free to contribute to this project and help improve it. Raise issues if you find any bugs or have any suggestions. Make pull requests to help develop it further and fix bugs. Also, you can feel free to contact me at my email bhavesh.laddagiri1@gmail.com or [instagram](https://www.instagram.com/nerdimite/) or [facebook](https://www.facebook.com/bhavesh.laddagiri)

<a name="ref"/>

### References
* Pose Estimation Code https://github.com/ildoonet/tf-pose-estimation/
