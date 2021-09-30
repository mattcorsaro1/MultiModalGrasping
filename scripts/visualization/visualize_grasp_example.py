'''
Copyright, 2021, Matt Corsaro, matthew_corsaro@brown.edu
'''

# Run this script to visualize a particular grasp pose.

import argparse
import numpy as np
import open3d as o3d
import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
from data_preparer import DataPreparer

# Given a 4x4 transformation matrix, create coordinate frame mesh at the pose
#     and scale down.
def o3dTFAtPose(pose, scale_down=10):
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    scaling_maxtrix = np.ones((4,4))
    scaling_maxtrix[:3, :3] = scaling_maxtrix[:3, :3]/scale_down
    scaled_pose = pose*scaling_maxtrix
    axes.transform(scaled_pose)
    return axes

# Convert (nx3) numpy array to Open3D point cloud
def npCloudToOpen3DCloud(cloud_np, color=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_np)
    if color is not None:
        cloud.colors = o3d.utility.Vector3dVector(color)
    return cloud

# Visualize the full combined cloud in the world frame (red),
#     the cloud, downsampled and cropped based on the grasp pose (green),
#     world axis frame, and grasp pose frame (4x4 transformation matrix)
def visualizeFullCloudAndGraspPose(cloud_full_np, cloud_sampled_cropped_np, pose):
    # Red
    cloud_full = npCloudToOpen3DCloud(cloud_full_np, np.array([[1., 0, 0] for i in range(cloud_full_np.shape[0])]))
    # Green
    cloud_sampled_cropped = npCloudToOpen3DCloud(cloud_sampled_cropped_np, np.array([[0, 1., 0] for i in range(cloud_sampled_cropped_np.shape[0])]))
    pose_axis = o3dTFAtPose(pose, scale_down=100)
    world_axis = o3dTFAtPose(np.eye(4))
    o3d.visualization.draw_geometries([cloud_sampled_cropped, cloud_full, world_axis, pose_axis])

def visualizeCroppedCloudAtGraspPose(cloud_np):
    cloud = npCloudToOpen3DCloud(cloud_np)
    pose_axis = o3dTFAtPose(np.eye(4), scale_down=100)
    o3d.visualization.draw_geometries([cloud, pose_axis])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset parameters
    parser.add_argument("--dir_path", type=str, default="../../data/labeled_examples/provided_dataset/", \
        help="Path to directory containing object sub-directories, each of which contains grasp pose and label files.")
    parser.add_argument("--bigbird_cloud_path", type=str, default="../../data/clouds/", \
        help="Path to directory containing object sub-directories, each of which contains calibrated BigBIRD point clouds.")

    parser.add_argument('--pointcloud_size', type=int, default=1024)

    parser.add_argument('--object_name', type=str, default="tapatio_hot_sauce")
    parser.add_argument('--camera', type=int, default=2, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--view', type=int, default=0, choices=[i*3 for i in range(120)])
    parser.add_argument('--grasp_index', type=int, default=0)

    args = parser.parse_args()

    data_preparer = DataPreparer(args.dir_path, args.bigbird_cloud_path, args.pointcloud_size)
    example = data_preparer.exampleTupleToPath((args.object_name, args.camera, args.view, args.grasp_index))
    cloud_1_filepath, cloud_2_filepath, handpose, cloud_1_indices, cloud_2_indices = example
    float_handpose = [float(strpose) for strpose in handpose.split(' ')]

    np_cloud_sampled_cropped = data_preparer.cloud_loader.NUMPY_readSample([cloud_1_filepath, cloud_2_filepath], [cloud_1_indices, cloud_2_indices])
    np_cloud_full = data_preparer.cloud_loader.NUMPY_read([cloud_1_filepath, cloud_2_filepath])
    print("Loaded sampled cloud of shape", np_cloud_sampled_cropped.shape, "and full cloud of shape", np_cloud_full.shape)

    transformed_cloud_np = data_preparer.cloud_loader.NUMPY_transformCloud(np_cloud_sampled_cropped, float_handpose)

    # Note that, as elsewhere, we use convention where z-axis (blue) is approach,
    #     y-axis (green) is closing direction, and x-axis (red) is cross product.
    # Invert again because this function inverts the matrix used for transformation.
    grasp_pose_world_frame = np.linalg.inv(data_preparer.cloud_loader.NUMPY_transformation_matrix_from_handpose(float_handpose))

    visualizeFullCloudAndGraspPose(np_cloud_full, np_cloud_sampled_cropped, grasp_pose_world_frame)
    visualizeCroppedCloudAtGraspPose(transformed_cloud_np)
