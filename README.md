# Updates

## 2021/09/30
Added a visualization script: scripts/visualization/visualize_grasp_example.py
Requires open3d (pip install).
Opens a window showing the full specified cloud and the cloud 54 degrees away from it in red, the cloud cropped and sampled in green, and the grasp pose, all in the world frame. After closing this window, shows the cropped and downsampled cloud in the grasp frame.

Usage example:

`python visualize_grasp_example.py --object_name tapatio_hot_sauce --camera 2 --view 0 --grasp_index 0`

Camera, as elsewhere, is an int between 1 and 5 representing the BigBIRD camera index, where 1 is horizontal, 5 is top-down, and 2-4 range between them. 
View, as elsewhere, is an int divisible by 3 between 0 and 357, representing rotation of turntable when capturing BigBIRD point cloud. See BigBIRD dataset for more info.
Grasp index is an integer representing the grasp to visualize. The maximum varies per cloud, and some clouds have no viewable grasps. 0 seems to correspond to grasps towards the top front left. Play around with this value to visualize different grasps; 500 is a reasonable guess for a grasp in the middle of an object.

# MultiModalGrasping
Code for Learning to Detect Multi-Modal Grasps for Dexterous Grasping in Dense Clutter

This branch contains a script to train a network to predict whether a grasp of each of n types would succeed at a pose given a cropped point cloud centered at that pose.

As the dataset this network was originally trained on stems from the BigBIRD dataset, this branch expects a dataset in a specific format.

To use your own dataset of grasp poses, labels, and point clouds without modifying code, the filepath layout and filename format must be copied exactly. See the provided dataset for an example, and dataset section for a more detailed explanation.

# Usage
The only currently runable script in this repository is train_pointconv.py. See the args in this script's main function. Some important ones are:

"--dir_path"
    directory containing pose and label files on which to train.
"--bigbird_cloud_path"
    directory containing calibrated BigBIRD point clouds, stored as npy files. Scripts to generate these files from the raw BigBIRD dataset will be released later.
"--output_dir"
    directory in which output is written.
"--label_to_train_with"
    "all" - 5 outputs, one per grasp type.
    "basic_both" - 2 outputs: basic precision and basic power.
    one of the five grasp types - 1 output.

Other important scripts are TensorFlowModule.py (where training operations are defined) and DeepLearningModule.py (where plotting functionality is defined.)
DeepLearningModule.py is particularly messy..

# Dataset
Download our set of calibrated BigBIRD pointclouds from [here](https://drive.google.com/file/d/1nWRRlS9kC7Gq0ueU8QyRx0NqTygyDFAK). Unzip this file so each object subdirectory is located in ./data/clouds.

## Point Clouds
Each object subdirectory contains a clouds directory. In each of these are 600 point clouds of the form Nn_m.npy, where n is an integer between 1 and 5 and m is an integer, divisible by 3, between 0 and 357. n is the BigBIRD camera number and m is the view number, the turntable orientation. See the BigBIRD dataset for more information. This code assumes that for each cloud, a cloud 54 degrees away also exists.

## Grasp Poses and Labels
Download our set of grasp poses and labels on the subset of included BigBIRD objects from [here](https://drive.google.com/file/d/17817NtNR6Tlhg2_SsbF-NDmYHWJwg0ja). Unzip this file so each object subdirectory is located in ./data/labeled_examples/provided_dataset.

Each object subdirectory contains:
    - A handposes directory, with files of grasp candidate poses of the form n_m.txt corresponding to each cloud. Each line encodes a pose in position-quaternion format.
    - A labels directory, with one file per handpose (and point cloud) file. Each line contains 5 grasp labels. If 0, the corresponding grasp type failed. If 1, it succeeded. If another integer, some error occured. 6 in particular means that drake crashed because the grasp failed, so we count this as label 0. We throw out all lines/grasp pose candidates with any value other than 0, 1, and 6 during training.

# Installation
The main dependency of this library is PointConv. We use the default release of PointConv with TensorFlow 1.14.

First, install Nvidia drivers, CUDA, and other dependencies of TensorFlow 1.14.

Set up a virtual environment and install TensorFlow 1.14.

Follow the instructions in the [PointConv repository](https://github.com/DylanWusee/pointconv) to install the TensorFlow ops. A copy of this code is included in scripts/PointNet2/tf_ops; compile these ops there, with either the PointConv compilation scripts or by modifying our compilation scripts. 

A more detailed write-up of installation steps will be included with the full release. If you have trouble or questions, please create an issue.

Other pip packages installed in our virtual environment, many of which may not be necessary, are:

absl-py==0.9.0
astor==0.8.1
backports.functools-lru-cache==1.6.1
backports.weakref==1.0.post1
cycler==0.10.0
enum34==1.1.10
funcsigs==1.0.2
futures==3.3.0
gast==0.3.3
google-pasta==0.2.0
grpcio==1.30.0
h5py==2.10.0
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.2
kiwisolver==1.1.0
Markdown==3.1.1
matplotlib==2.2.4
mock==3.0.5
numpy==1.16.6
opencv-python==4.2.0.32
pandas==0.24.2
Pillow==6.2.2
pkg-resources==0.0.0
protobuf==3.12.2
pyparsing==2.4.7
pypcd==0.1.1
python-dateutil==2.8.1
python-lzf==0.2.4
pytz==2020.1
scikit-learn==0.20.4
scipy==1.2.3
six==1.15.0
subprocess32==3.5.4
tensorboard==1.14.0
tensorflow-estimator==1.14.0
tensorflow-gpu==1.14.0
termcolor==1.1.0
Werkzeug==1.0.1
wrapt==1.12.1
