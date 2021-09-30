'''
Copyright, 2021, Matt Corsaro, matthew_corsaro@brown.edu
'''

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from PointConv import feature_encoding_layer

def placeholder_inputs(batch_size, num_point, num_pointcloud_channels=3, num_labels=1):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    if num_pointcloud_channels != 3:
        sys.exit("This code was not tested on inputs with more or less than 3 channels. Try at your own risk.")
    labels_pl = None
    if num_labels > 1:
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_labels))
    else:
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None, num_class=2, sigma=0.05, weight_decay=None, num_labels=1):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    
    l1_xyz, l1_points = feature_encoding_layer(point_cloud, point_cloud, npoint=512, radius = 0.1, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[64,64,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius = 0.4, sigma = 4 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3_points, npoint=36, radius = 0.8, sigma = 8 * sigma, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')

    net = tf.reshape(l4_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc5', bn_decay=bn_decay)
    if num_labels > 1:
        net = tf_util.fully_connected(net, num_class*num_labels, bn=True, is_training=is_training, scope='fc6', bn_decay=bn_decay)
        net = tf.reshape(net, [batch_size, num_labels, num_class])
    else:
        net = tf_util.fully_connected(net, num_class, bn=True, is_training=is_training, scope='fc6', bn_decay=bn_decay)

    softmax_not_for_training = tf.nn.softmax(net)
    return net, softmax_not_for_training, end_points


def get_loss(pred, label, end_points):
    """ pred: BxNxC,
        label: BxN"""
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss
