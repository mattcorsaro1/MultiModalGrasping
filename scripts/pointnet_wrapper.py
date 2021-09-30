'''
Copyright, 2021, Matt Corsaro, matthew_corsaro@brown.edu
'''

import sys
import os

import importlib
import tensorflow as tf

# Input is point clouds
# Adapted from PointNet2's train.py - github.com/charlesq34/pointnet2
class PointNet:
    def __init__(self, robotiq_label, batch_size, min_num_points, num_pointcloud_channels, base_learning_rate, decay_step, decay_rate, \
        bn_init_decay, bn_decay_decay_step, bn_decay_decay_rate, bn_decay_clip, decay_learning_rate=False, num_labels=1):

        print "PARAMS:"
        print "robotiq_label", robotiq_label
        print "batch_size", batch_size
        print "min_num_points", min_num_points
        print "num_pointcloud_channels", num_pointcloud_channels
        print "base_learning_rate", base_learning_rate
        print "decay_step", decay_step
        print "decay_rate", decay_rate
        print "bn_init_decay", bn_init_decay
        print "bn_decay_decay_step", bn_decay_decay_step
        print "bn_decay_decay_rate", bn_decay_decay_rate
        print "bn_decay_clip", bn_decay_clip
        print "decay_learning_rate", decay_learning_rate
        print "num_labels", num_labels

        self.robotiq_label = robotiq_label
        self.batch_size = batch_size
        self.min_num_points = min_num_points
        self.num_pointcloud_channels = num_pointcloud_channels
        self.base_learning_rate = base_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.bn_init_decay = bn_init_decay
        self.bn_decay_decay_step = bn_decay_decay_step
        self.bn_decay_decay_rate = bn_decay_decay_rate
        self.bn_decay_clip = bn_decay_clip
        self.num_labels = num_labels

        # import network module
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(base_dir)
        sys.path.append(os.path.join(base_dir, 'PointNet2/models'))
        self.model = importlib.import_module("pointconv_weight_density_n16")

        pointclouds_pl, labels_pl = self.model.placeholder_inputs(self.batch_size, self.min_num_points, self.num_pointcloud_channels, \
            num_labels=self.num_labels)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment the 'batch' parameter
        # for you every time it trains.
        batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
        bn_decay = self.get_bn_decay(batch)
        # Get model and loss
        pred, softmax, end_points = self.model.get_model(pointclouds_pl, is_training_pl, \
            bn_decay=bn_decay, num_labels=self.num_labels)
        self.model.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        # Get training operator
        learning_rate = self.base_learning_rate
        if decay_learning_rate:
            # TODO: Unclear if this is actually updated..
            learning_rate = self.get_learning_rate(batch)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=batch)

        self.ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'softmax': softmax,
               'loss': total_loss,
               'train_op': train_op,
               'step': batch,
               'end_points': end_points}

    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
            self.base_learning_rate,  # Base learning rate.
            batch * self.batch_size,  # Current index into the dataset.
            self.decay_step,          # Decay step.
            self.decay_rate,          # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
        return learning_rate        

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
                          self.bn_init_decay,
                          batch*self.batch_size,
                          self.bn_decay_decay_step,
                          self.bn_decay_decay_rate,
                          staircase=True)
        bn_decay = tf.minimum(self.bn_decay_clip, 1 - bn_momentum)
        return bn_decay