#!/usr/bin/env python

'''
Copyright, 2021, Matt Corsaro, matthew_corsaro@brown.edu
'''

import matplotlib
matplotlib.use('Agg')
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import time

import sklearn.metrics
from sklearn.utils.fixes import signature

def countLabels(labels):
    counter = Counter(labels)
    num_pos_ex = counter[1]
    num_neg_ex = counter[0]
    return (num_pos_ex, num_neg_ex)

class DeepLearningModule(object):
    def __init__(self, train_data_paths, train_labels, test_data_paths, test_labels, output_path, num_labels):
        self.obj_type_dict = {"3m_high_tack_spray_adhesive": "cylinder", \
            "advil_liqui_gels": "box", \
            "bai5_sumatra_dragonfruit": "cylinder", \
            "band_aid_clear_strips": "box", \
            "band_aid_sheer_strips": "box", \
            "campbells_soup_at_hand_creamy_tomato": "cylinder", \
            "canon_ack_e10_box": "box", \
            "cheez_it_white_cheddar": "box", \
            "chewy_dipps_chocolate_chip": "box", \
            "cholula_chipotle_hot_sauce": "other", \
            "cinnamon_toast_crunch": "box", \
            "clif_crunch_chocolate_chip": "box", \
            "clif_z_bar_chocolate_chip": "box", \
            "coffee_mate_french_vanilla": "cylinder", \
            "colgate_cool_mint": "other", \
            "crayola_24_crayons": "box", \
            "crest_complete_minty_fresh": "other", \
            "crystal_hot_sauce": "cylinder", \
            "detergent": "other", \
            "dove_beauty_cream_bar": "box", \
            "eating_right_for_healthy_living_blueberry": "box", \
            "fruit_by_the_foot": "box", \
            "gushers_tropical_flavors": "box", \
            "haagen_dazs_butter_pecan": "cylinder", \
            "hersheys_cocoa": "box", \
            "honey_bunches_of_oats_with_almonds": "box", \
            "hunts_paste": "cylinder", \
            "hunts_sauce": "cylinder", \
            "ikea_table_leg_blue": "cylinder", \
            "krylon_crystal_clear": "cylinder", \
            "krylon_low_odor_clear_finish": "cylinder", \
            "krylon_short_cuts": "cylinder", \
            "mom_to_mom_butternut_squash_pear": "other", \
            "mom_to_mom_sweet_potato_corn_apple": "other", \
            "motts_original_assorted_fruit": "box", \
            "nature_valley_crunchy_oats_n_honey": "box", \
            "nice_honey_roasted_almonds": "cylinder", \
            "nutrigrain_apple_cinnamon": "box", \
            "nutrigrain_toffee_crunch_chocolatey_toffee": "box", \
            "pepto_bismol": "cylinder", \
            "pop_secret_butter": "box", \
            "pop_secret_light_butter": "box", \
            "pop_tarts_strawberry": "box", \
            "pringles_bbq": "cylinder", \
            "progresso_new_england_clam_chowder": "cylinder", \
            "quaker_big_chewy_chocolate_chip": "box", \
            "quaker_big_chewy_peanut_butter_chocolate_chip": "box", \
            "red_bull": "cylinder", \
            "ritz_crackers": "box", \
            "softsoap_gold": "other", \
            "softsoap_white": "other", \
            "south_beach_good_to_go_peanut_butter": "box", \
            "spam": "box", \
            "spongebob_squarepants_fruit_snaks": "box", \
            "suave_sweet_guava_nectar_body_wash": "other", \
            "sunkist_fruit_snacks_mixed_fruit": "box", \
            "tapatio_hot_sauce": "other", \
            "v8_fusion_peach_mango": "cylinder", \
            "v8_fusion_strawberry_banana": "cylinder", \
            "vo5_tea_therapy_healthful_green_tea_smoothing_shampoo": "other", \
            "white_rain_sensations_apple_blossom_hydrating_body_wash": "other", \
            "white_rain_sensations_ocean_mist_hydrating_body_wash": "other", \
            "zilla_night_black_heat": "box"}
        self.obj_types = list(set(self.obj_type_dict.values()))
        # List of data paths and labels
        self.train_data_paths = train_data_paths
        self.train_labels = train_labels
        self.test_data_paths = test_data_paths
        self.test_labels = test_labels
        self.data_processing_time = 0

        # Directory in which to save plots
        self.output_path = output_path

        self.process_obj_distributions()

        self.num_labels = num_labels
        self.grasp_types = ["wide_power", "wide_precision", "basic_power", "basic_precision", "pincher"]

        # Set/appended to during training, used for plotting
        self.train_loss_per_epoch = []

        self.train_time_per_epoch = []
        self.test_time_per_epoch = []

        self.train_acc_per_epoch = []
        self.test_acc_per_epoch = []
        self.train_pos_acc_per_epoch = []
        self.train_neg_acc_per_epoch = []
        self.test_pos_acc_per_epoch = []
        self.test_neg_acc_per_epoch = []
        self.test_accs_per_objclass_per_epoch = {obj_type: [] for obj_type in self.obj_types}

        self.multi_label_train_num_pos_ex, self.multi_label_train_num_neg_ex = [], []
        self.multi_label_test_num_pos_ex, self.multi_label_test_num_neg_ex = [], []
        if self.num_labels > 1:
            for label_type_i in range(len(self.train_labels[0])):
                type_train_labels = [label[label_type_i] for label in self.train_labels]
                type_test_labels = [label[label_type_i] for label in self.test_labels]
                type_train_pos, type_train_neg = countLabels(type_train_labels)
                type_test_pos, type_test_neg = countLabels(type_test_labels)
                self.multi_label_train_num_pos_ex.append(type_train_pos)
                self.multi_label_train_num_neg_ex.append(type_train_neg)
                self.multi_label_test_num_pos_ex.append(type_test_pos)
                self.multi_label_test_num_neg_ex.append(type_test_neg)
        else:
            self.single_label_train_num_pos_ex, self.single_label_train_num_neg_ex = countLabels(self.train_labels)
            self.single_label_test_num_pos_ex, self.single_label_test_num_neg_ex = countLabels(self.test_labels)

        self.train_acc_txt_filename = self.output_path + "/train_accs.txt"
        self.test_acc_txt_filename = self.output_path + "/test_accs.txt"

        self.latest_train_net_prob_out = None
        self.latest_test_net_prob_out = None
        self.latest_train_labels = None
        self.RM_latest_test_labels = None

    def process_train_epoch(self):
        epoch_acc, pos_acc, neg_acc = None, None, None
        if self.num_labels > 1:
            epoch_acc, pos_acc, neg_acc = [], [], []
            for grasp_type_i in range(len(self.latest_train_labels[0])):
                type_latest_train_labels = [type_label[grasp_type_i] for type_label in self.latest_train_labels]
                type_latest_train_net_prob_out = [type_output[grasp_type_i] for type_output in self.latest_train_net_prob_out]
                type_epoch_acc, type_pos_acc, type_neg_acc = self.calc_totalacc_pos_and_neg(type_latest_train_labels, type_latest_train_net_prob_out)
                epoch_acc.append(type_epoch_acc)
                pos_acc.append(type_pos_acc)
                neg_acc.append(type_neg_acc)
        else:
            epoch_acc, pos_acc, neg_acc = self.calc_totalacc_pos_and_neg(self.latest_train_labels, self.latest_train_net_prob_out)
        self.train_acc_per_epoch.append(epoch_acc)
        self.train_pos_acc_per_epoch.append(pos_acc)
        self.train_neg_acc_per_epoch.append(neg_acc)
        print "Latest train accuracy:", epoch_acc
        self.train_acc_txt_file = open(self.train_acc_txt_filename, 'a')
        if self.num_labels > 1:
            epoch_acc_str = ""
            for acc in epoch_acc:
                epoch_acc_str += str(acc)
            epoch_acc_str = epoch_acc_str[:-1]
            self.train_acc_txt_file.write(str(epoch_acc_str) + '\n')
        else:
            self.train_acc_txt_file.write(str(epoch_acc) + '\n')
        self.train_acc_txt_file.close()

    def process_test_epoch(self):
        def compute_precision_recall_f1(labels, probs):
            FP, TP, FN, TN = 0.0, 0.0, 0.0, 0.0
            for i in range(len(labels)):
                if labels[i] < 0.5 and probs[i] >= 0.5:
                    FP += 1
                elif labels[i] >= 0.5 and probs[i] >= 0.5:
                    TP += 1
                elif labels[i] >= 0.5 and probs[i] < 0.5:
                    FN += 1
                elif labels[i] < 0.5 and probs[i] < 0.5:
                    TN += 1
            prec = 0 if (TP+FP == 0) else TP/(TP+FP)
            rec = 0 if (TP+FN == 0) else TP/(TP+FN)
            f1 = 0 if (prec+rec == 0) else 2*prec*rec/(prec+rec)
            return [prec, rec, f1]

        if len(self.test_labels) != len(self.RM_latest_test_labels):
            print "Something is seriously wrong, different number of test grasp labels."
        else:
            num_off = 0
            for i in range(len(self.test_labels)):
                if self.test_labels[i] != self.RM_latest_test_labels[i]:
                    num_off += 1
            print "Labels off by", num_off, "of", len(self.test_labels)
        if self.test_labels != self.RM_latest_test_labels:
            print "GRASP LABEL VECTORS ARE NOT THE SAME", len(self.test_labels), len(self.RM_latest_test_labels), self.test_labels[0], self.RM_latest_test_labels[0], self.test_labels[-1], self.RM_latest_test_labels[-1]
        else:
            print "Grasp label vectors are equal, both of size", len(self.test_labels)

        epoch_acc, pos_acc, neg_acc = None, None, None
        if self.num_labels > 1:
            epoch_acc, pos_acc, neg_acc = [], [], []
            tot_prec, tot_rec, tot_f1 = 0.0, 0.0, 0.0
            print "Grasp type prec, rec, f1:",
            for grasp_type_i in range(len(self.test_labels[0])):
                type_test_labels = [type_label[grasp_type_i] for type_label in self.test_labels]
                type_latest_test_net_prob_out = [type_output[grasp_type_i] for type_output in self.latest_test_net_prob_out]
                type_epoch_acc, type_pos_acc, type_neg_acc = self.calc_totalacc_pos_and_neg(type_test_labels, type_latest_test_net_prob_out)
                epoch_acc.append(type_epoch_acc)
                pos_acc.append(type_pos_acc)
                neg_acc.append(type_neg_acc)

                prec, rec, f1 = compute_precision_recall_f1(type_test_labels, type_latest_test_net_prob_out)
                print "(", prec, rec, f1, "),",
                tot_prec += prec
                tot_rec += rec
                tot_f1 += f1
            print
            print "Avg prec, rec, f1:", tot_prec/len(self.test_labels[0]), tot_rec/len(self.test_labels[0]), tot_f1/len(self.test_labels[0])
        else:
            #if 
            epoch_acc, pos_acc, neg_acc = self.calc_totalacc_pos_and_neg(self.test_labels, self.latest_test_net_prob_out)
            self.update_test_acc_per_epoch_dict()
            print "Prec, rec, f1:", compute_precision_recall_f1(self.test_labels, self.latest_test_net_prob_out)
        self.test_acc_per_epoch.append(epoch_acc)
        self.test_pos_acc_per_epoch.append(pos_acc)
        self.test_neg_acc_per_epoch.append(neg_acc)
        print "Latest test accuracy:", epoch_acc
        self.test_acc_txt_file =  open(self.test_acc_txt_filename, 'a')
        if self.num_labels > 1:
            epoch_acc_str = ""
            for acc in epoch_acc:
                epoch_acc_str += str(acc)
            epoch_acc_str = epoch_acc_str[:-1]
            self.test_acc_txt_file.write(str(epoch_acc_str) + '\n')
        else:
            self.test_acc_txt_file.write(str(epoch_acc) + '\n')
        self.test_acc_txt_file.close()
        # return whether test acc is max
        if self.num_labels > 1:
            avg_test_accs = [sum(accs)/len(accs) for accs in self.test_acc_per_epoch]
            return avg_test_accs[-1] == max(avg_test_accs)
        else:
            return self.test_acc_per_epoch[-1] == max(self.test_acc_per_epoch)

    def process_obj_distributions(self):
        def getFiletype(filepath):
            return filepath.split('.')[-1]

        # Arg: List of image paths for one example
        def name_from_img_paths(img_paths_one_ex):
            first_path = img_paths_one_ex[0]
            # Assume /path/to/bigbird/obj/clouds/NP#_###.npy
            return first_path.replace('//', '/').split('/')[-3]

        # Name of object w/ robot suffix for each example in set
        train_objct_names = [name_from_img_paths(example) for example in self.train_data_paths]
        test_objct_names = [name_from_img_paths(example) for example in self.test_data_paths]

        # Used for plotting train/test distribution
        self.train_objct_dist = Counter(train_objct_names)
        self.test_objct_dist = Counter(test_objct_names)
        self.plot_obj_dists()

        # For per-obj-class accuracy plotting
        self.test_obj_names = [name for name in test_objct_names]
        self.test_ex_per_obj_class = {key: 0 for key in self.obj_types}
        for objct in self.test_objct_dist:
            self.test_ex_per_obj_class[self.obj_type_dict[objct]] += self.test_objct_dist[objct]

    def update_test_acc_per_epoch_dict(self):
        correct_per_obj_class = {obj_class: 0 for obj_class in self.obj_types}
        for i in xrange(len(self.test_labels)):
            #if abs(self.test_labels[i]-self.latest_test_net_prob_out[i]) < 0.5:
            if (self.test_labels[i]==0 and self.latest_test_net_prob_out[i] < 0.5) or (self.test_labels[i]==1 and self.latest_test_net_prob_out[i] >= 0.5):
                correct_per_obj_class[self.obj_type_dict[self.test_obj_names[i]]] += 1
        for obj_class in self.obj_types:
            self.test_accs_per_objclass_per_epoch[obj_class].append(0 if self.test_ex_per_obj_class[obj_class]==0 else float(correct_per_obj_class[obj_class])/self.test_ex_per_obj_class[obj_class])

    def calc_totalacc_pos_and_neg(self, labels, net_prob_out):
        #the count of true negatives is c_0_0, false negatives is c_1_0, true positives is c_1_1, and false positives is c_0_1
        confusion_matrix = sklearn.metrics.confusion_matrix([int(l) for l in labels], [0 if i<0.5 else 1 for i in net_prob_out])
        true_negatives, false_positives, false_negatives, true_positives = confusion_matrix.ravel()
        total_acc = 0 if (true_positives + true_negatives + false_positives + false_negatives == 0) else float(true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
        pos_acc = 0 if (true_positives+false_negatives == 0) else float(true_positives)/(true_positives+false_negatives)
        neg_acc = 0 if (true_negatives+false_positives == 0) else float(true_negatives)/(true_negatives+false_positives)
        return (total_acc, pos_acc, neg_acc)

    def update_plots(self):
        if len(self.latest_test_net_prob_out)==0:
            print "CANNOT PLOT WITHOUT NETWORK OUTPUT"
            return

        #TODO(mcorsaro): do this for multilabel..
        if self.num_labels == 1:
            #### ROC, Prec vs. acc
            self.latest_false_pos_rates, self.latest_true_pos_rates, thresholds = sklearn.metrics.roc_curve(self.test_labels, self.latest_test_net_prob_out, pos_label=1)
            self.precision, self.recall, _ = sklearn.metrics.precision_recall_curve(self.test_labels, self.latest_test_net_prob_out)
            self.average_precision = sklearn.metrics.average_precision_score(self.test_labels, self.latest_test_net_prob_out)
            # Only regenerate these plots while test accuracy improves, or every time for multi_label
            if self.test_acc_per_epoch[-1] == max(self.test_acc_per_epoch):
                epoch = len(self.test_acc_per_epoch)
                self.plot_roc(epoch)
                self.plot_prec_vs_acc(epoch)

        ## Acc vs epoch
        if self.num_labels > 1:
            multi_label_train_accs, multi_label_test_accs = [], []
            grasp_type_indices = None
            grasp_types = self.grasp_types
            if self.num_labels == 2:
                grasp_type_indices = range(len(self.test_labels[0]))
                grasp_types = [type_i for type_i in self.grasp_types if "basic" in type_i]
            elif self.num_labels == 5:
                grasp_type_indices = range(5)
            for grasp_i in grasp_type_indices:
                type_train_accs = [acc[grasp_i] for acc in self.train_acc_per_epoch]
                type_train_accs_pos = [acc[grasp_i] for acc in self.train_pos_acc_per_epoch]
                type_train_accs_neg = [acc[grasp_i] for acc in self.train_neg_acc_per_epoch]
                type_test_accs = [acc[grasp_i] for acc in self.test_acc_per_epoch]
                type_test_accs_pos = [acc[grasp_i] for acc in self.test_pos_acc_per_epoch]
                type_test_accs_neg = [acc[grasp_i] for acc in self.test_neg_acc_per_epoch]
                self.plot_accs_train_test_pos_neg(self.output_path + "/accuracies_" + grasp_types[grasp_i] + ".jpg", \
                    type_train_accs, type_train_accs_pos, type_train_accs_neg, type_test_accs, type_test_accs_pos, type_test_accs_neg, \
                    self.multi_label_train_num_pos_ex[grasp_i], self.multi_label_train_num_neg_ex[grasp_i], \
                    self.multi_label_test_num_pos_ex[grasp_i], self.multi_label_test_num_neg_ex[grasp_i], ' ' + grasp_types[grasp_i])
                multi_label_train_accs.append(type_train_accs)
                multi_label_test_accs.append(type_test_accs)
            avg_train_acc_per_epoch, avg_test_acc_per_epoch = [], []
            for epoch in range(len(self.train_acc_per_epoch)):
                avg_train_acc_per_epoch.append(sum(self.train_acc_per_epoch[epoch])/len(self.train_acc_per_epoch[epoch]))
                avg_test_acc_per_epoch.append(sum(self.test_acc_per_epoch[epoch])/len(self.test_acc_per_epoch[epoch]))
            multi_label_train_accs.append(avg_train_acc_per_epoch)
            multi_label_test_accs.append(avg_test_acc_per_epoch)
            self.plot_multi_label_accs(self.output_path + "/accuracies.jpg", multi_label_train_accs, multi_label_test_accs)
        else:
            self.plot_accs_train_test_pos_neg(self.output_path + "/accuracies.jpg", self.train_acc_per_epoch, self.train_pos_acc_per_epoch, \
                self.train_neg_acc_per_epoch, self.test_acc_per_epoch, self.test_pos_acc_per_epoch, self.test_neg_acc_per_epoch, \
                self.single_label_train_num_pos_ex, self.single_label_train_num_neg_ex, self.single_label_test_num_pos_ex, self.single_label_test_num_neg_ex)
            # TODO(mcorsaro): plot accuracies per obj class with multi_label
            self.plot_test_accs_per_obj_class()
        self.plot_training_loss()
        self.plot_times()

    def formatInitAndMaxAcc(self, accuracies):
        init_acc_str = "{0:.3g}".format(accuracies[0])
        max_acc = max(accuracies)
        max_acc_str = "{0:.3g}".format(max_acc)
        max_acc_index = accuracies.index(max_acc)
        return 'Init: ' + init_acc_str + ', max: ' + max_acc_str + ' at ' + str(max_acc_index)

    def plot_multi_label_accs(self, accplot_filename, train_accuracies_list, test_accuracies_list):
        color_list = None
        grasp_types_used = None
        if self.num_labels == 5:
            color_list = ['blue', 'orange', 'magenta', 'purple', 'cyan', 'green']
            grasp_types_used = self.grasp_types
        elif self.num_labels == 2:
            color_list = ['magenta', 'purple', 'green']
            grasp_types_used = [gtype for gtype in self.grasp_types if "basic" in gtype]
        train_legend_entries = grasp_types_used + ["Avg"]
        test_legend_entries = grasp_types_used + ["Avg"]
        y_axis_values = ["Train acc (" + self.formatInitAndMaxAcc(train_accuracies_list[-1]) + ')', \
            "Test acc (" + self.formatInitAndMaxAcc(test_accuracies_list[-1]) + ')']
        titles = ["Train accuracy vs. epoch", "Test accuracy vs. epoch"]
        plot_and_save_train_and_test_accs(train_accuracies_list, test_accuracies_list, color_list, \
            train_legend_entries, test_legend_entries, y_axis_values, titles, accplot_filename)

    def plot_accs_train_test_pos_neg(self, accplot_filename, train_accs, train_pos_accs, train_neg_accs, test_accs, test_pos_accs, test_neg_accs, \
        train_num_pos, train_num_neg, test_num_pos, test_num_neg, additional_title=''):
        train_accuracies_list = [train_accs, train_pos_accs, train_neg_accs]
        test_accuracies_list = [test_accs, test_pos_accs, test_neg_accs]
        color_list = ['blue', 'green', 'red']

        legend_entries = lambda num_positive, num_negative : ['All %d' % (num_positive+num_negative), \
            'Pos %d' % num_positive, 'Neg %d' % num_negative]
        train_legend_entries = legend_entries(train_num_pos, train_num_neg)
        test_legend_entries = legend_entries(test_num_pos, test_num_neg)

        # One for train, one for test
        y_axis_values = ["Train acc (" + self.formatInitAndMaxAcc(train_accs) + ')', \
            "Test acc (" + self.formatInitAndMaxAcc(test_accs) + ')']
        titles = ["Train accuracy vs. epoch" + additional_title, "Test accuracy vs. epoch" + additional_title]

        plot_and_save_train_and_test_accs(train_accuracies_list, test_accuracies_list, color_list, \
            train_legend_entries, test_legend_entries, y_axis_values, titles, accplot_filename)

    def plot_test_accs_per_obj_class(self):
        if len(self.test_acc_per_epoch) > 1:
            save_path = self.output_path + "/accuracies_per_class.jpg"
            plt.figure(figsize=(10,8))
            for obj_class in self.test_accs_per_objclass_per_epoch:
                assert(len(self.test_acc_per_epoch) == len(self.test_accs_per_objclass_per_epoch[obj_class]))

            lw=2
            init_acc_str = "{0:.3g}".format(self.test_acc_per_epoch[0])
            max_acc = max(self.test_acc_per_epoch)
            max_acc_str = "{0:.3g}".format(max_acc)
            max_acc_index = self.test_acc_per_epoch.index(max_acc)
            endx = len(self.test_acc_per_epoch)-1

            plt.xlim([0, endx])
            plt.ylim([0.0, 1.05])
            plt.plot([0, endx], [1, 1], color='darkorange', lw=lw, linestyle='--')

            for obj_class in self.test_accs_per_objclass_per_epoch:
                plt.plot(range(len(self.test_accs_per_objclass_per_epoch[obj_class])), self.test_accs_per_objclass_per_epoch[obj_class], lw=lw, label='{}: {}'.format(obj_class, self.test_ex_per_obj_class[obj_class]))

            plt.plot(range(len(self.test_acc_per_epoch)), self.test_acc_per_epoch, lw=lw, label="Total: {}".format(self.single_label_test_num_neg_ex + self.single_label_test_num_pos_ex))
            plt.xlabel('Epoch')
            plt.ylabel('acc (Avg init: ' + init_acc_str + ', max: ' + max_acc_str + ' at ' + str(max_acc_index) + ')')
            plt.title('Accuracy per object class vs. epoch')
            plt.legend(loc="lower right")
            plt.tight_layout()

            plt.savefig(save_path)
            plt.close()

    def plot_roc(self, epoch):
        if self.latest_false_pos_rates.shape[0] > 1 and self.latest_true_pos_rates.shape[0] > 1:
            save_path = self.output_path + "/roc_curve.jpg"
            roc_auc = sklearn.metrics.auc(self.latest_false_pos_rates, self.latest_true_pos_rates)
            plt.figure()
            lw=2
            plt.plot(self.latest_false_pos_rates, self.latest_true_pos_rates, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.plot([0, 0], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.plot([0, 1], [1, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            title = "ROC at epoch {}".format(epoch)
            plt.title(title)
            plt.legend(loc="lower right")
            plt.savefig(save_path)
            plt.close()
        else:
            print "At least 2 points are needed to compute area under curve. You gave:", self.true_pos_rates.shape, self.false_pos_rates.shape, self.true_pos_rates, self.false_pos_rates

    def plot_prec_vs_acc(self, epoch):
        save_path = self.output_path + "/prec_vs_acc.jpg"
        plt.figure(figsize=(20,10))
        xmax = 1.05
        ymax = 1.05

        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(self.recall, self.precision, color='b', alpha=0.2, where='post')
        plt.fill_between(self.recall, self.precision, alpha=0.2, color='b', **step_kwargs)

        precisions_to_check = [0.8, 0.9, 0.95, 0.97, 0.99]
        recalls_at_precisions = dict()

        def get_recall_at_given_prec(precisions, recalls, desired_prec):
            desired_recall_index, closest_prec = min(enumerate(precisions), key=lambda x:abs(x[1]-desired_prec))
            return (closest_prec, recalls[desired_recall_index])

        for prec2check in precisions_to_check:
            closest_prec, recall_at = get_recall_at_given_prec(self.precision, self.recall, prec2check)
            recalls_at_precisions[closest_prec] = recall_at

        for i, prec2plot in enumerate(sorted(recalls_at_precisions.keys())):
            prec_plot_loc = float(i+1)/(len(recalls_at_precisions)+1)
            rec_plot_loc = prec_plot_loc-0.03

            plt.vlines(x=recalls_at_precisions[prec2plot], ymin=0, ymax=ymax, color='black')
            plt.text(recalls_at_precisions[prec2plot]+0.01, prec_plot_loc, "p %0.2f" % prec2plot)
            plt.text(recalls_at_precisions[prec2plot]+0.01, rec_plot_loc, "r %0.2f" % recalls_at_precisions[prec2plot])

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, ymax])
        plt.xlim([0.0, xmax])
        plt.title('2-class Precision-Recall curve at epoch {}: AP={:0.2f}'.format(epoch, self.average_precision))

        plt.savefig(save_path)
        plt.close()

    def plot_vs_epoch(self, data, name, units=None, info=None):
        if len(data) > 1:
            save_path = self.output_path + '/' + name.replace(' ', '_').lower() + "_per_epoch.jpg"
            plt.figure(figsize=(20,10))
            plt.plot(range(len(data)), data, color='blue', lw=2, label=name)
            plt.xlim([0.0, len(data)-1])
            plt.ylim([0.0, max(data)+1])
            plt.xlabel('Epoch')
            ylabel = name
            if units != None:
                ylabel += " (" + units + ')'
            plt.ylabel(ylabel)
            plot_title = name + " vs. Epoch"
            if info != None:
                plot_title += "    " + info
            plt.title(plot_title)
            plt.legend(loc="lower right")
            plt.savefig(save_path)
            plt.close()

    def plot_training_loss(self):
        print "Latest training loss:", self.train_loss_per_epoch[-1]
        self.plot_vs_epoch(self.train_loss_per_epoch, "Average batch training loss")

    def plot_times(self):
        self.plot_vs_epoch(self.train_time_per_epoch, "Training time", units="minutes", info="Preprocessing time: {0:.2f}".format(self.data_processing_time))
        self.plot_vs_epoch(self.test_time_per_epoch, "Testing time", units="minutes")

    def plot_obj_dists(self):
        save_path = self.output_path + "/train_test_dist.jpg"
        objct_list = list(set(self.train_objct_dist.keys() + self.test_objct_dist.keys()))
        objs = [objct for objct in objct_list]
        objs.sort()

        to_plot = dict()
        for obj_type in self.obj_types:
            to_plot[obj_type + "_train"] = []
            to_plot[obj_type + "_test"] = []
            to_plot[obj_type + "_name"] = []
        for obj in self.obj_type_dict:
            objct = obj
            if objct in self.train_objct_dist or objct in self.test_objct_dist:
                to_plot[self.obj_type_dict[obj] + "_train"].append(0 if objct not in self.train_objct_dist else self.train_objct_dist[objct])
                to_plot[self.obj_type_dict[obj] + "_test"].append(0 if objct not in self.test_objct_dist else self.test_objct_dist[objct])
                to_plot[self.obj_type_dict[obj] + "_name"].append(obj)

        train_list = []
        test_list = []
        name_list = []
        obj_name_plot_x = []
        current_obj_count = 0
        for obj_type in self.obj_types:
            if len(to_plot[obj_type + "_name"]) != 0:
                train_list += to_plot[obj_type + "_train"]
                test_list += to_plot[obj_type + "_test"]
                name_list += to_plot[obj_type + "_name"]
                train_list.append(0)
                test_list.append(0)
                name_list.append("")
                obj_name_plot_x.append((obj_type, len(to_plot[obj_type + "_name"])/2 + current_obj_count))
                current_obj_count += len(to_plot[obj_type + "_name"]) + 1

        plt.figure(figsize=(20,10))
        assert(len(train_list) == len(test_list) and len(train_list) == len(name_list))
        max_x = len(name_list)
        max_y = max([train_list[i] + test_list[i] for i in range(len(name_list))])

        ind = np.arange(max_x)    # the x locations for the groups
        width = 0.9  # 1 = touching

        trainplot = plt.bar(ind, train_list, width)
        testplot = plt.bar(ind, test_list, width, bottom=train_list)

        for i in range(len(name_list)):
            label = name_list[i] + "     train: " + str(train_list[i]) + "  test: " + str(test_list[i]) if name_list[i] != "" else ""
            plt.text(i, max_y/100, label, color='black', rotation="vertical", horizontalalignment='center', verticalalignment='bottom')#, fontweight='bold')
        for obj_type, x_pos in obj_name_plot_x:
            plt.text(x_pos, -1*max_y/30, obj_type, color='black')#, rotation="vertical", horizontalalignment='center', verticalalignment='bottom')#, fontweight='bold')

        plt.ylabel('Number of examples per object')
        plt.title('Train/Test split per object')
        #plt.xticks(ind, name_list, rotation="vertical")
        plt.xticks([])
        plt.legend((trainplot[0], testplot[0]), ('Train', 'Test'))
        plt.savefig(save_path)
        plt.close()

def acc_subplot(ax, accuracies_list, colors, legend_entries, yaxis, title):
    lw=2
    endx = max(len(accuracies) for accuracies in accuracies_list)-1

    ax.set_xlim([0, endx])
    ax.set_ylim([0.0, 1.05])
    ax.plot([0, endx], [1, 1], color='darkorange', lw=lw, linestyle='--')
    for line_i in range(len(colors)):
        ax.plot(range(len(accuracies_list[line_i])), accuracies_list[line_i], color=colors[line_i], lw=lw, label=legend_entries[line_i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel(yaxis)
    ax.set_title(title)
    ax.legend(loc="lower right")

def plot_and_save_train_and_test_accs(train_accuracies_list, test_accuracies_list, color_list, train_legend_entries, \
    test_legend_entries, y_axis_values, titles, save_path):
    # Each of the first 5 lists has an entry for each line to plot. Length must be equal.
    # y values and titles: list of length 2; one for train, one for test

    if len(color_list) != len(train_accuracies_list) or \
        len(color_list) != len(test_accuracies_list) or \
        len(color_list) != len(train_legend_entries) or \
        len(color_list) != len(test_legend_entries) or \
        len(color_list) == 0:

        print "Length of values to plot of are of different sizes, not generating a plot."
        print len(train_accuracies_list), len(test_accuracies_list), len(color_list), len(train_legend_entries), \
            len(test_legend_entries)
        throw()
    for i in range(len(color_list)):
        if len(train_accuracies_list[i]) <= 1 or len(test_accuracies_list[i]) <= 1:
            # Only plot if each will have at least 2 values for each
            return

    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(20,8))

    acc_subplot(axs[0], train_accuracies_list, color_list, train_legend_entries, y_axis_values[0], titles[0])
    acc_subplot(axs[1], test_accuracies_list, color_list, test_legend_entries, y_axis_values[1], titles[1])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
