# Virtex
## End-to-End Pose Recognition

Virtex is an easy to use API for pose recognition which lets you estimate and classify the poses.
You can train on images having those poses or you can also train for poses in real time using the webcam. The classifier
implements XG Boost Classifier by default but can be changed to your own classification model.

## Table of Contents

* [Installation](#install)
* [Documentation (API)](#docs)
    * [Data Generation](#gen)
    * [Preprocessing](#pre)
    * [Training](#train)
    * [Prediction](#pred)
 * [Documentation (CLI)](#docs)
    * [Data Generation](#gen)
    * [Preprocessing](#pre)
    * [Training](#train)
    * [Prediction](#pred)
* [References](#ref)

<a name="install"/>

### Installation

Clone the repo and install 3rd-party libraries.

```bash
$ git clone https://github.com/Nerdimite37/virtex.git
$ cd virtex
$ pip3 install -r requirements.txt
```
Build c++ library for post processing. See : https://github.com/Nerdimite37/virtex/tree/master/tf_pose/pafprocess
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```
Alternatively, you can install this repo as a shared package using pip.

```bash
$ git clone https://github.com/Nerdimite37/virtex.git
$ cd virtex
$ python setup.py install  # Or, `pip install -e .`
```

<a name="ref"/>

### Reference
* Pose Estimation Code https://github.com/ildoonet/tf-pose-estimation/
