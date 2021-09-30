'''
Copyright, 2021, Matt Corsaro, matthew_corsaro@brown.edu
'''

from cloud_loader import CloudLoader
import numpy as np

# Read a file df, split by parsechar, remove extra lines at the end.
def parse_datafile(df, parsechar):
    data_file = open(df, 'r')
    data_lines = data_file.read().split(parsechar)
    if data_lines == [''] or len(data_lines) == 0:
        return []
    while data_lines[-1] == '' and len(data_lines) > 0:
        data_lines.pop()
    return data_lines

# Read a cloud file and return shape of cloud.
def getCloudSize(cloud_path):
    np_cloud = np.load(cloud_path)
    return np_cloud.shape[0]

# Helper class for reading examples.
class DataPreparer(object):
    def __init__(self, dir_path, calibrated_bigbird_cloud_dir, min_num_pointnet_points):
        self.dir_path = dir_path
        self.calibrated_bigbird_cloud_dir = calibrated_bigbird_cloud_dir
        self.min_num_pointnet_points = min_num_pointnet_points
        self.cloud_loader = CloudLoader(self.min_num_pointnet_points, 3)

    # example_tuple of the form (object_name, camera, view, grasp_index)
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
            indices_1, indices_2 = self.cloud_loader.NUMPY_read_and_select_indices([cloud_file, second_cloud_file], [float(strpose) for strpose in this_cam_view_handposes[grasp_index].split(' ')])
            if len(indices_1) != 0 and len(indices_2) !=0 :
                example = [cloud_file, second_cloud_file, this_cam_view_handposes[grasp_index], indices_1, indices_2]
                return example
        return None