#!/usr/bin/env python

'''
Copyright, 2021, Matt Corsaro, matthew_corsaro@brown.edu
'''

from DeepLearningModule import DeepLearningModule

import time
import numpy as np
import math
import sys
import os

import tensorflow as tf

import cloud_loader
import pointnet_wrapper

class TensorFlowModule(DeepLearningModule):
    def __init__(self, train_data_paths, train_labels, test_data_paths, test_labels, learning_rate, batch_size, output_path, pointcloud_size, num_epochs, num_cores, label_to_train_with, save_model=True):
        num_labels = 5 if label_to_train_with=="all" else (2 if label_to_train_with=="basic_both" else 1)
        super(TensorFlowModule, self).__init__(train_data_paths, train_labels, test_data_paths, test_labels, output_path, num_labels)

        print "Using TensorFlow", tf.__version__
        print "GPU Available:", tf.test.is_gpu_available()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.min_num_points = pointcloud_size
        self.num_epochs = num_epochs
        self.label_to_train_with = label_to_train_with

        self.train_steps_per_epoch=int(math.ceil(len(self.train_labels)/float(self.batch_size)))
        self.test_steps_per_epoch=int(math.ceil(len(self.test_labels)/float(self.batch_size)))

        self.save_model = save_model

        ################################################### DATASETS ###################################################
        data_read_start_time=time.time()

        self.cloud_loader = cloud_loader.CloudLoader(self.min_num_points, 3)

        # ['pc1_filepath', 'pc2_filepath', 'pose', [indices_1], [indices_2]]
        train_data_paths = [paths_and_handpose[:2] for paths_and_handpose in self.train_data_paths]
        test_data_paths = [paths_and_handpose[:2] for paths_and_handpose in self.test_data_paths]
        train_handposes_float = [[float(i) for i in paths_and_handpose[2].split(' ')] for paths_and_handpose in self.train_data_paths]
        test_handposes_float = [[float(i) for i in paths_and_handpose[2].split(' ')] for paths_and_handpose in self.test_data_paths]
        train_cloud_indices = None
        test_cloud_indices = None
        assert(len(self.train_data_paths[0]) == 5)
        train_cloud_indices = [paths_and_handpose[3:5] for paths_and_handpose in self.train_data_paths]
        test_cloud_indices = [paths_and_handpose[3:5] for paths_and_handpose in self.test_data_paths]

        train_path_label_ds = tf.data.Dataset.from_tensor_slices((train_data_paths, train_handposes_float, train_cloud_indices, self.train_labels))
        test_path_label_ds = tf.data.Dataset.from_tensor_slices((test_data_paths, test_handposes_float, test_cloud_indices, self.test_labels))

        train_path_label_ds = train_path_label_ds.apply(tf.data.experimental.shuffle_and_repeat(len(self.train_labels), num_epochs))
        test_path_label_ds = test_path_label_ds.repeat(count=num_epochs)
        
        self.train_ds = train_path_label_ds.map(self.cloud_loader.load_and_preprocess_cloud_from_path_label, num_parallel_calls=num_cores)
        self.test_ds = test_path_label_ds.map(self.cloud_loader.load_and_preprocess_cloud_from_path_label, num_parallel_calls=num_cores)
        #Also you shouldn't be prefetching THAT many batches. Prefetch 10 at most. If your parallelism works properly that's all you should need. It's a waste of RAM of otherwise.
        self.train_ds = self.train_ds.batch(self.batch_size).prefetch(25)
        self.test_ds = self.test_ds.batch(self.batch_size).prefetch(25)

        print "Created tf.Dataset from paths, overall runtime so far is", (time.time()-data_read_start_time)/60, "minutes."
        self.data_processing_time = (time.time() - data_read_start_time)/60

        ################################################ NET AND SESSION ###############################################
        print "Creating net"
        decay_step = 200000
        decay_rate = 0.7
        bn_init_decay = 0.5
        bn_decay_decay_step = float(decay_step)
        bn_decay_decay_rate = 0.5
        bn_decay_clip = 0.99

        self.net = pointnet_wrapper.PointNet(self.batch_size, self.min_num_points, 3, self.learning_rate, decay_step, decay_rate, \
            bn_init_decay, bn_decay_decay_step, bn_decay_decay_rate, bn_decay_clip, num_labels=self.num_labels)

        # Create Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.session = tf.Session(config=config)
        # Initialize all Variables
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

        if self.save_model:
            self.saver = tf.train.Saver()

        ############################################## DATASET ITERATORS ###############################################
        self.train_iter = tf.data.Iterator.from_structure(self.train_ds.output_types, self.train_ds.output_shapes)
        self.next_train_element = self.train_iter.get_next()
        train_init_op = self.train_iter.make_initializer(self.train_ds)
        self.session.run(train_init_op)

        self.test_iter = tf.data.Iterator.from_structure(self.test_ds.output_types, self.test_ds.output_shapes)
        self.next_test_element = self.test_iter.get_next()
        test_init_op = self.test_iter.make_initializer(self.test_ds)
        self.session.run(test_init_op)

        print "TensorFlow Learner created."

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print "Total number of trainable parameters in the network:", total_parameters

    ################################################# EPOCH-LEVEL EVAL #################################################
    def train_epoch(self):
        batch_losses = []
        shuffled_labels = []
        train_label_probs = []
        train_epoch_start_time = time.time()

        for batch_num in range(self.train_steps_per_epoch):
            data, labels = self.session.run(self.next_train_element)
            loss, softmax_out = self.train_pointnet_batch(data, labels)

            shuffled_labels += labels.tolist()
            positive_label_prob = None
            if self.num_labels > 1:
                positive_label_prob = (softmax_out[:,:,1]).tolist()
            else:
                positive_label_prob = (softmax_out[:,1]).tolist()
            train_label_probs += positive_label_prob
            batch_losses.append(loss)

        self.train_loss_per_epoch.append(sum(batch_losses)/len(batch_losses) if len(batch_losses)>0 else 0)
        self.latest_train_net_prob_out = train_label_probs
        self.latest_train_labels = shuffled_labels
        # TODO(mcorsaro): refactor
        self.process_train_epoch()
        self.train_time_per_epoch.append((time.time()-train_epoch_start_time)/60)

    def test_epoch(self):
        test_label_probs = []
        RM_latest_test_labels = []
        test_epoch_start_time = time.time()

        time_gathering = 0
        time_running = 0
        for batch_num in range(self.test_steps_per_epoch):
            softmax_out = None
            batch_start = time.time()
            data, labels = self.session.run(self.next_test_element)
            process_start = time.time()
            softmax_out = self.test_pointnet_batch(data, labels)
            process_end = time.time()
            time_gathering += (process_start-batch_start)
            time_running += (process_end-process_start)

            RM_latest_test_labels += labels.tolist()
            positive_label_prob = None
            if self.num_labels > 1:
                positive_label_prob = (softmax_out[:,:,1]).tolist()
            else:
                positive_label_prob = (softmax_out[:,1]).tolist()
            test_label_probs += positive_label_prob

        epoch_test_time = (time.time()-test_epoch_start_time)/60
        epoch_gather_time = time_gathering/60
        epoch_run_time = time_running/60
        print "Epoch", len(self.train_loss_per_epoch), "testing time", epoch_test_time, "min total for", len(test_label_probs), "examples.", epoch_gather_time, "to gather data,", epoch_run_time, "to pass through net."
        self.test_time_per_epoch.append(epoch_test_time)

        self.latest_test_net_prob_out = test_label_probs
        self.RM_latest_test_labels = RM_latest_test_labels
        # TODO(mcorsaro): refactor
        test_acc_is_max = self.process_test_epoch()

        if self.save_model and test_acc_is_max:
            epoch_num = len(self.train_loss_per_epoch)-1
            model_file_name = self.output_path + "/tf_model_epoch_" + str(epoch_num)
            self.saver.save(self.session, model_file_name)

    ################################################# BATCH-LEVEL EVAL #################################################
    def train_pointnet_batch(self, data, labels):
        is_training = True
        feed_dict = {self.net.ops['pointclouds_pl']: data,
                     self.net.ops['labels_pl']: labels,
                     self.net.ops['is_training_pl']: is_training,}
        step, _, loss_val, pred_val, softmax_out = self.session.run([
            self.net.ops['step'],
            self.net.ops['train_op'],
            self.net.ops['loss'],
            self.net.ops['pred'],
            self.net.ops['softmax']], feed_dict=feed_dict)
        return (loss_val, softmax_out)

    def test_pointnet_batch(self, data, labels):
        is_training = False
        feed_dict = {self.net.ops['pointclouds_pl']: data,
                     self.net.ops['labels_pl']: labels,
                     self.net.ops['is_training_pl']: is_training}
        step, loss_val, pred_val, softmax_out = self.session.run([
            self.net.ops['step'],
            self.net.ops['loss'],
            self.net.ops['pred'],
            self.net.ops['softmax']], feed_dict=feed_dict)
        return softmax_out
