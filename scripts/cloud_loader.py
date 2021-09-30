'''
Copyright, 2021, Matt Corsaro, matthew_corsaro@brown.edu
'''

import math

import numpy as np

import tensorflow as tf

class CloudLoader:
    def __init__(self, min_num_points, num_pointcloud_channels):
        self.min_num_points = min_num_points
        self.num_pointcloud_channels = num_pointcloud_channels

    def load_and_preprocess_cloud_from_path_label(self, cloud_paths, handpose, indices, label):
        return self.load_and_preprocess_and_transform_cloud(cloud_paths, handpose, indices), label

    def load_and_preprocess_and_transform_cloud(self, cloud_paths, handpose, indices):
        # cloud_paths - Tensor or numpy array (not sure) containing two paths to .npy point cloud files as strings,
        # handpose - tensor/numpy array/list of floats representing transform - x y z position, w x y z quaternion rotation
        tf_cloud = tf.py_func(self.NUMPY_readSampleTransform, [cloud_paths, handpose, indices], tf.float32)
        return tf_cloud

    def NUMPY_readSampleTransform(self, cloud_paths, handpose, indices):
        combined_cloud = self.NUMPY_readSample(cloud_paths, indices)
        transformed_cloud = self.NUMPY_transformCloud(combined_cloud, handpose)
        return transformed_cloud.astype(np.float32)

    def NUMPY_read(self, cloud_paths):
        cloud_1 = np.load(cloud_paths[0])
        cloud_2 = np.load(cloud_paths[1])
        combined_cloud = np.concatenate((cloud_1, cloud_2))
        if self.num_pointcloud_channels == 3:
            combined_cloud = combined_cloud[:,:3]
        return combined_cloud.astype(np.float32)

    def NUMPY_readSample(self, cloud_paths, indices):
        cloud_1 = np.load(cloud_paths[0])
        cloud_2 = np.load(cloud_paths[1])
        sampled_cloud_1, sampled_cloud_2 = None, None
        if len(indices) == 0:
            sampled_cloud_1 = self.NUMPY_downsampleSingleCloud(cloud_1)
            sampled_cloud_2 = self.NUMPY_downsampleSingleCloud(cloud_2)
        else:
            cloud_1_indices = indices[0]
            cloud_2_indices = indices[1]
            sampled_cloud_1 = cloud_1[cloud_1_indices, :]
            sampled_cloud_2 = cloud_2[cloud_2_indices, :]
        combined_cloud = np.concatenate((sampled_cloud_1, sampled_cloud_2))
        if self.num_pointcloud_channels == 3:
            combined_cloud = combined_cloud[:,:3]
        return combined_cloud.astype(np.float32)

    def NUMPY_transformCloud(self, cloud, handpose):
        trans_mat = self.NUMPY_transformation_matrix_from_handpose(handpose)
        transformed_cloud = self.transformCloud(cloud[:,:3], trans_mat)
        return transformed_cloud

    def transformCloud(self, cloud, transform):
        # Input cloud is nx3
        # 3xn
        transpose_cloud = np.transpose(cloud)
        # 4xn cloud matrix by adding extra row of 1s
        transpose_cloud_and_ones = np.concatenate((transpose_cloud, np.ones((1, transpose_cloud.shape[1]))), axis=0)
        # Transform
        transformed_transpose_cloud_and_ones = np.matmul(transform, transpose_cloud_and_ones)
        # 3xn - remove the ones
        transformed_transpose_cloud = np.delete(transformed_transpose_cloud_and_ones, (3), axis=0)
        # nx3 transpose back
        transformed_cloud = np.transpose(transformed_transpose_cloud)
        return transformed_cloud

    def NUMPY_transformation_matrix_from_handpose(self, handpose_tensor):
        #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
        w = handpose_tensor[3]
        x = handpose_tensor[4]
        y = handpose_tensor[5]
        z = handpose_tensor[6]

        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w

        xy = x * y
        zw = z * w
        xz = x * z
        yw = y * w
        yz = y * z
        xw = x * w

        trans_mat_row_0 = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), handpose_tensor[0]]
        trans_mat_row_1 = [2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), handpose_tensor[1]]
        trans_mat_row_2 = [2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2, handpose_tensor[2]]
        trans_mat_row_3 = [0., 0., 0., 1.]

        trans_mat = np.array([trans_mat_row_0, trans_mat_row_1, trans_mat_row_2, trans_mat_row_3])
        return np.linalg.inv(trans_mat)

    def NUMPY_downsampleSingleCloud(self, cloud):
        valid_indices = np.arange(cloud.shape[0])
        np.random.shuffle(valid_indices)
        return cloud[valid_indices[:self.min_num_points/2], :]

    def NUMPY_read_and_select_indices(self, cloud_paths, handpose_float, crop=True):
        cloud_1 = np.load(cloud_paths[0])
        cloud_2 = np.load(cloud_paths[1])
        tf_cloud_1 = self.NUMPY_transformCloud(cloud_1, handpose_float)
        tf_cloud_2 = self.NUMPY_transformCloud(cloud_2, handpose_float)
        if crop:
            # Note that the tf_cloud_1[:,2] > -0.085 condition was not added until later.
            # However, < 40 points (before downsampling) from 7 of 37000 clouds in our dataset are affected
            tf_cloud_1 = tf_cloud_1[np.logical_and(\
                np.logical_and(\
                np.logical_and(\
                    np.abs(tf_cloud_1[:,0]) < 0.09, \
                    np.abs(tf_cloud_1[:,1]) < 0.076), \
                    tf_cloud_1[:,2] < 0.155),\
                tf_cloud_1[:,2] > -0.085)]
            tf_cloud_2 = tf_cloud_2[np.logical_and(\
                np.logical_and(\
                np.logical_and(\
                    np.abs(tf_cloud_2[:,0]) < 0.09, \
                    np.abs(tf_cloud_2[:,1]) < 0.076), \
                    tf_cloud_2[:,2] < 0.155),\
                tf_cloud_2[:,2] > -0.085)]
        return (self.NUMPY_select_random_indices(tf_cloud_1).tolist(), self.NUMPY_select_random_indices(tf_cloud_2).tolist())

    def NUMPY_select_random_indices(self, cloud):
        new_cloud_num_points = int(self.min_num_points/2)
        valid_indices = np.arange(cloud.shape[0])

        if valid_indices.shape[0] == 0:
            #print "No graspable points."
            return np.array([], dtype=np.int32)
        elif valid_indices.shape[0] < new_cloud_num_points:
            # number of times to repeat the small cloud so we can upsample from it
            num_repeats = int(math.ceil(float(new_cloud_num_points)/valid_indices.shape[0]))
            return np.tile(valid_indices, num_repeats)[:new_cloud_num_points]
        else:
            np.random.shuffle(valid_indices)
            return valid_indices[:new_cloud_num_points]
