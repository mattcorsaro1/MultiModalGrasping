#!/usr/bin/env python

'''
Copyright, 2021, Matt Corsaro, matthew_corsaro@brown.edu
'''

# Run this script to train a PointConv network to predict grasp success
#     for any set of grasp types given a point cloud and grasp pose.

import argparse
import datetime
import numpy as np
import random
import os
import sys
import time

from cloud_loader import CloudLoader
from TensorFlowModule import TensorFlowModule

def read_multimodal_label_file(label_dir, obj_name, cam, view, unlabeled_label=9, num_modes=5):
    """
    Reads specified file and returns dictionary mapping object, camera, view,
        and index to a label list.

    @type   label_dir: string
    @param  label_dir: path to an object's label directory.

    @type   obj_name: obj_name
    @param  obj_name: object name.

    @type   cam: int
    @param  cam: BigBIRD camera index: integer between 1 and 5, inclusive.

    @type   view: int
    @param  view: BigBIRD view index: integer between 0 and 357, inclusive.
                 Divisible by 3.

    @type   unlabeled_label: int
    @param  unlabeled_label: If all grasp types at a grasp pose have this label,
                             a grasp was not attempted. Do not include in dataset.

    @type   num_modes: int
    @param  num_modes: Number of grasp types at dataset generation time, not
                       number of grasp types used to train the network

    @type   grasp_to_label_dict: dict
    @return grasp_to_label_dict: dictionary mapping tuples of:
                                 (object name, camera, view, index) to list of
                                 num_modes integers, each either 0 or 1.
    """

    grasp_to_label_dict = {}
    filepath = label_dir + '/' + str(cam) + '_' + str(view) + ".txt"
    if os.path.isfile(filepath):
        # Read specified label file
        data_lines = parse_datafile(filepath, '\n')

        # Empty
        if len(data_lines) == 0:
            return grasp_to_label_dict

        # A pose could be labeled with num_modes failure labels if it wasn't
        #     attempted because of something like a detected collision.
        # E.g. when unlabeled_label == 9 and num_modes == 5, examples with the
        #     label "9 9 9 9 9".
        unlabeled_string = ''
        for seg in [str(unlabeled_label) + ' ' for i in range(num_modes-1)] + [str(unlabeled_label)]:
            unlabeled_string += seg
        
        # Iterate over grasp pose. A number of grasp poses are attempted on each
        #     point cloud.
        for i, label_str in enumerate(data_lines):
            # Skip this example if it's unlabeled.
            if label_str == unlabeled_string:
                pass
            else:
                label_ex = True
                # if Drake crashed because obj hit floor (label 6),
                #     consider it a failure (label 0).
                label_str = label_str.replace('6', '0')
                # If any other error code was thrown during any grasp type attempt
                #     (collision, etc.) at this pose, do not use the pose.
                for tossable_label in range(9)[2:]:
                    if str(tossable_label) in label_str:
                        label_ex = False
                if label_ex:
                    # Add this grasp pose to the dataset
                    label_list = label_str.split()
                    label_list_int = [int(lab) for lab in label_list]
                    grasp_to_label_dict[(obj_name, cam, view, i)] = label_list_int
    return grasp_to_label_dict

def read_multimodal_label_files(data_path):
    """
    Reads all label files in given path and returns dictionary mapping
        object, camera, view, and index to a label list.

    @type   data_path: string
    @param  data_path: Directory in which dataset is stored. Object
                       sub-directories are located here.

    @type   grasp_to_label_dict: dict
    @return grasp_to_label_dict: dictionary mapping tuples of:
                                 (object name, camera, view, index) to list of
                                 num_modes integers, each either 0 or 1.
    """
    obj_label_dict = {}
    objects = os.listdir(data_path)
    for obj in objects:
        label_dir = data_path + '/' + obj + "/labels"
        if not os.path.isdir(label_dir):
            print "COULD NOT FIND", label_dir
            continue
        # Based on expected BigBIRD dataset structure
        for cam in range(6)[1:]:
            for view in [3*v for v in range(120)]:
                label_dict = read_multimodal_label_file(label_dir, obj, cam, view)
                obj_label_dict.update(label_dict)
    return obj_label_dict

# Read a file df, split by parsechar, remove extra lines at the end.
def parse_datafile(df, parsechar):
    data_file = open(df, 'r')
    data_lines = data_file.read().split(parsechar)
    if data_lines == [''] or len(data_lines) == 0:
        return []
    while data_lines[-1] == '' and len(data_lines) > 0:
        data_lines.pop()
    return data_lines

# Split dataset dset_list into training and testing subsets.
def train_test_split(dset_list, percent_train):
    # Assumes list has been shuffled
    train = dset_list[:int(len(dset_list)*(percent_train/100.))]
    test = dset_list[int(len(dset_list)*(percent_train/100.)):]
    return train, test

# Read a cloud file and return shape of cloud.
def getCloudSize(cloud_path):
    np_cloud = np.load(cloud_path)
    return np_cloud.shape[0]

# Shuffle unshuffled_list. If a random seed is given, use it for this shuffle, then revert back to a random random seed
def shuffle_list_with_seed(unshuffled_list, random_seed=None):
    new_random_seed = random.randint(0,99999999)
    if random_seed != None:
        random.seed(random_seed)
    random.shuffle(unshuffled_list)
    if random_seed != None:
        random.seed(new_random_seed)

# Helper class for reading examples.
class DataPreparer(object):
    def __init__(self, dir_path, calibrated_bigbird_cloud_dir, min_num_pointnet_points):
        self.dir_path = dir_path
        self.calibrated_bigbird_cloud_dir = calibrated_bigbird_cloud_dir
        self.min_num_pointnet_points = min_num_pointnet_points

    def exampleTupleToPath(self, example_tuple):
        object_name, cam, view, grasp_index = example_tuple
        cloud_file = self.calibrated_bigbird_cloud_dir + '/' + object_name + "/clouds/NP" + str(cam) + "_" + str(view) + ".npy"
        second_cloud_num = (view + 54) % 360
        second_cloud_file = self.calibrated_bigbird_cloud_dir + '/' + object_name + "/clouds/NP" + str(cam) + "_" + str(second_cloud_num) + ".npy"

        # Don't use any examples from this cloud set if the combined clouds don't have enough points for PointNet
        # Though we could make sure the sum of the number of points in the two clouds is at least min_num_pointnet_points,
        # ensure both clouds have sufficient number of points
        if getCloudSize(cloud_file) >= self.min_num_pointnet_points/2 and getCloudSize(second_cloud_file) >= self.min_num_pointnet_points/2:
            handpose_dir = self.dir_path + "/" + object_name + "/handposes/"
            handpose_file = handpose_dir + '/' + str(cam) + '_' + str(view) + ".txt"
            this_cam_view_handposes = parse_datafile(handpose_file, '\n')
            indices_1, indices_2 = cloud_loader.NUMPY_read_and_select_indices([cloud_file, second_cloud_file], [float(strpose) for strpose in this_cam_view_handposes[grasp_index].split(' ')])
            if len(indices_1) != 0 and len(indices_2) !=0 :
                example = [cloud_file, second_cloud_file, this_cam_view_handposes[grasp_index], indices_1, indices_2]
                return example
        return None

# Return lists of training and testing example paths and labels, unbatched
def drakeFilepath2datasetpaths(dir_path, calibrated_bigbird_cloud_dir, \
    cloud_loader, \
    batch_size, percent_train, random_seed, \
    split_by_obj, min_num_pointnet_points):

    #(obj_name, cam, view, i) -> label list
    obj_label_dict = read_multimodal_label_files(dir_path)

    train_dict_keys, test_dict_keys = [], []
    if split_by_obj:
        # Do not share objects between training and testing sets.
        objects = os.listdir(dir_path)
        shuffle_list_with_seed(objects, random_seed)
        num_train_objs = int(len(objects)*percent_train/100.)
        train_objs, test_objs = objects[:num_train_objs], objects[num_train_objs:]
        for key in obj_label_dict:
            if key[0] in train_objs:
                train_dict_keys.append(key)
            elif key[0] in test_objs:
                test_dict_keys.append(key)
            else:
                print "Object is somehow neither in training or testing set:", key[0]
                sys.exit()
    else:
        # Don't select obj+cam pairs to use for train test, select 85/15 across the whole set
        label_dict_keys = obj_label_dict.keys()
        shuffle_list_with_seed(label_dict_keys, random_seed)
        train_dict_keys, test_dict_keys = train_test_split(label_dict_keys, percent_train)

    if len(train_dict_keys) == 0 or len(test_dict_keys) == 0:
        print "Not enought data."
        sys.exit()

    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []
    data_preparer = DataPreparer(dir_path, calibrated_bigbird_cloud_dir, min_num_pointnet_points)
    # Each key is a tuple of the form (obj, cam, view, grasp)
    # Convert to examples of the form
    #     [point cloud path 1, point cloud path 2, pose, cloud 1 indices, cloud 2 indices]
    for train_key in train_dict_keys:
        train_example = data_preparer.exampleTupleToPath(train_key)
        if train_example is not None:
            train_paths.append(train_example)
            train_labels.append(obj_label_dict[train_key])
    for test_key in test_dict_keys:
        test_example = data_preparer.exampleTupleToPath(test_key)
        if test_example is not None:
            test_paths.append(test_example)
            test_labels.append(obj_label_dict[test_key])

    # Get rid of some data so it fits in batches
    num_train_ex_to_use = len(train_labels) - (len(train_labels) % batch_size)
    num_test_ex_to_use = len(test_labels) - (len(test_labels) % batch_size)
    train_paths = train_paths[:num_train_ex_to_use]
    train_labels = train_labels[:num_train_ex_to_use]
    test_paths = test_paths[:num_test_ex_to_use]
    test_labels = test_labels[:num_test_ex_to_use]
    return train_paths, train_labels, test_paths, test_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset parameters
    parser.add_argument("--dir_path", type=str, default="../data/labeled_examples/provided_dataset/", \
        help="Path to directory containing object sub-directories, each of which contains grasp pose and label files.")
    parser.add_argument("--bigbird_cloud_path", type=str, default="../data/clouds/", \
        help="Path to directory containing object sub-directories, each of which contains calibrated BigBIRD point clouds.")
    parser.add_argument("--output_dir", type=str, default="../output", \
        help="Directory in which to generate a time-stamped sub-directory to save model and output data and plots.")
    
    # Training parameters
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1.0e-5)
    parser.add_argument('--percent_train', type=int, default=85, \
        help="Percent of dataset to use for training. 100-percent_train percent will be used for validation.")

    parser.add_argument('--pointcloud_size', type=int, default=1024)

    # If split_by_obj, objects in training and testing sets are disjoint.
    # If split_random, examples are selected at random to include in training
    #     and testing sets. The same object could be found in both training and
    #     testing, though no particular example would be found in both.
    parser.add_argument('--split_by_obj', dest='split_by_obj', action='store_true')
    parser.add_argument('--split_random', dest='split_by_obj', action='store_false')
    parser.set_defaults(split_by_obj=False)

    # If all, network outputs 5 values and predicts whether all 5 grasps types
    #     would succeed (5Type). If basic_both, network outputs 2 values and
    #     predicts whether basic_precision and basic_power would succeed
    #     (2Type). Other options have network output one value and train on
    #     the one corresponding grasp type.
    parser.add_argument("--label_to_train_with", choices=['all', "wide_power", "wide_precision", "basic_power", "basic_precision", "pincher", "basic_both"], default="all")

    # Number of cores on your machine.
    parser.add_argument('--num_cores', type=int, default=12)

    args = parser.parse_args()

    data_read_start_time = time.time()

    # Generate a timestamped directory in which to save training results and
    #     models.
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    timestamped_output_directory = args.output_dir + '/' + timestamp
    os.makedirs(timestamped_output_directory)

    # Object for loading point clouds.
    cloud_loader = CloudLoader(args.pointcloud_size, 3)
    # Generates a dataset, split into training and validation sets, with example
    #     paths and labels.
    # One value in train_data_paths is a list of length 5 containing:
    #     Path to point cloud
    #     Path to point cloud 54 degrees away
    #     Pose - string with 3 position and 4 orientation values
    #     List of len(pointcloud_size) of indices to use from first point cloud
    #     List of len(pointcloud_size) of indices to use from second point cloud
    # One value in train_labels is a list of 5 integers
    train_data_paths, train_labels, test_data_paths, test_labels = \
        drakeFilepath2datasetpaths(args.dir_path, args.bigbird_cloud_path, \
            cloud_loader, \
            args.batch_size, args.percent_train, args.random_seed, \
            args.split_by_obj, args.pointcloud_size)

    # Ordered list of grasp types, order defined in data generation script.
    grasp_types = ["wide_power", "wide_precision", "basic_power", "basic_precision", "pincher"]
    
    # Now, select labels based on label_to_train_with
    if args.label_to_train_with == "all":
        # Keep all 5
        pass
    elif args.label_to_train_with == "basic_both":
        # Keep basic_power and basic_precision
        grasp_type_indices = [grasp_types.index(type_i) for type_i in grasp_types if "basic" in type_i]
        train_labels = [[ex_labels[type_i] for type_i in grasp_type_indices] for ex_labels in train_labels]
        test_labels = [[ex_labels[type_i] for type_i in grasp_type_indices] for ex_labels in test_labels]
    else:
        # Keep the one corresponding to single selected grasp type
        grasp_type_index = grasp_types.index(args.label_to_train_with)
        print(grasp_type_index)
        train_labels = [l[grasp_type_index] for l in train_labels]
        test_labels = [l[grasp_type_index] for l in test_labels]

    # Make sure each example has a label
    if (len(train_data_paths) != len(train_labels)):
        print len(train_data_paths), "training paths and", len(train_labels), "labels"
        sys.exit()
    if (len(test_data_paths) != len(test_labels)):
        print len(test_data_paths), "testing paths and", len(test_labels), "labels"
        sys.exit()

    print "Arranged filenames and labels in", (time.time() - data_read_start_time)/60, "minutes."
    print "NOW TRAINING WITH", len(train_labels), "TRAINING EXAMPLES AND", len(test_labels), "TEST EXAMPLES."

    learner = TensorFlowModule(train_data_paths, train_labels, test_data_paths, test_labels, args.learning_rate, \
        args.batch_size, timestamped_output_directory, args.pointcloud_size, args.num_epochs, \
        args.num_cores, args.label_to_train_with, save_model=True)

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        learner.train_epoch()
        learner.test_epoch()
        learner.update_plots()

        print "Epoch", epoch, "in", (time.time() - epoch_start_time)/60, "minutes."
