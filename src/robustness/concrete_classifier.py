# -*- coding: utf-8 -*-
# =============================================================================
# File: interval_classifier.py
# Updated: 05/11/2022
# =============================================================================
'''Implementation of the concrete k-NN classifier'''
# =============================================================================
# Dependencies:
#   ../base.py
#   ../dataset.py
#   ../utils/distances.py
#   ../utils/min_heap.py
# =============================================================================

from __future__ import annotations
from configparser import ConfigParser
from os import makedirs
from os.path import exists, join
from logging import basicConfig, getLogger
from itertools import combinations

from robustness import Dataset, Boolean, Integer, Literal, Map, Set, String, Real, Vector
from robustness.utils import compute_distance, MinHeap

class ConcreteClassifier:
    '''Represents a k-NN classifier'''

    def __init__(self) -> None:
        '''
        Lets the class initialize the object's attributes
        ''' 
        log_filename = '{}.log'.format(self.__class__.__name__)

        settings_parser = ConfigParser()
        settings_parser.read('settings.ini')

        if not exists(settings_parser.get('DEFAULT', 'logs_dir')):
            makedirs(settings_parser.get('DEFAULT', 'logs_dir'))

        basicConfig(
            filename=join(settings_parser.get('DEFAULT', 'logs_dir'), log_filename),
            format='%(message)s',
            filemode='w'
        )
        self.logger = getLogger()
        self.logger.setLevel(50)

    def __compute_distances(self,
        test_point,
        distance_metric: String
    ) -> MinHeap:
        '''
        Computes the abstract distances of each point in the training set to the adversarial region
        :param test_point: Target test point
        :param distance_metric: Metric to evaluate the distance between two points
        :return: The distances in a min heap structure
        '''
        distances = []
        for train_point, train_label in zip(self.__training_set.get_points(), self.__training_set.get_labels()):
            distances.append((
                compute_distance(test_point, train_point, distance_metric),
                train_label
            ))
        return MinHeap(distances)

    def get_training_set(self) -> Dataset:
        '''
        Returns the stored training set
        :return: Stored training set
        '''
        return self.__training_set

    def set_log(self,
        on: Boolean
    ) -> None:
        '''
        Allows to activate or deactivate the logs
        :param on: Whether they should be activated or not
        '''
        return self.logger.setLevel(50 * (not on))

    def fit(self,
        training_set: Dataset
    ) -> None:
        '''
        Fits the classifier from the training dataset
        :param trainig_set: Training set to fit
        '''
        self.__training_set = training_set

    def classify(self,
        test_point: Vector[Real],
        k_values: Vector[Integer],
        distance_metric: String = 'euclidean'
    ) -> Map[Set[Literal]]:
        '''
        Performs the abstract classification
        :param test_point: Point to be classified
        :param k_values: Number of neighbors to consider (one or more values)
        :param distance_metric: Metric to evaluate the abstract distance between two points
        :return: The most voted labels for each k
        '''
        self.logger.info('- test point: {}\n'.format(test_point))
        
        distances = self.__compute_distances(test_point, distance_metric)

        most_voted_labels = {}
        for k in k_values:     
            if distances.get_size() <= k or distances.get_nth_smallest(k)[0] < distances.get_nth_smallest(k + 1)[0]:
                scores = {label : 0 for label in self.__training_set.get_classes()}
                for i in range(1, k + 1):
                    scores[distances.get_nth_smallest(i)[1]] += 1
                scores = {label: score for label, score in scores.items() if score > 0}
                max_score = max(scores.values())
                most_voted_labels[k] = Set([label for label in scores if scores[label] == max_score])
                self.logger.info('\tk = {} -> scores: {} -> winning: {}'.format(k, scores, most_voted_labels[k]))

            else:
                # the h-th distance is equal to the k-th distance (holds) for some h > k
                kth_distance, kth_label = distances.get_nth_smallest(k)
                uncertain_labels = [kth_label]
                for i in range(k + 1, distances.get_size() + 1):
                    ith_distance, ith_label = distances.get_nth_smallest(i)
                    if ith_distance == kth_distance:
                        uncertain_labels.append(ith_label)
                    else:
                        break

                closest_labels = []
                for i in range(k - 1, 0, -1):
                    ith_distance, ith_label = distances.get_nth_smallest(i)
                    if ith_distance == kth_distance:
                        uncertain_labels.append(ith_label)
                    else:
                        for j in range(1, i + 1):
                            closest_labels.append(distances.get_nth_smallest(j)[1])
                        break
                
                most_voted_labels[k] = Set()
                
                # all possible ways to choose k points
                for possible_selection in Set(combinations(uncertain_labels, k - len(closest_labels))):
                    scores = {label : 0 for label in self.__training_set.get_classes()}
                    for label in closest_labels:
                        scores[label] += 1
                    for label in possible_selection:
                        scores[label] += 1
                    scores = {label: score for label, score in scores.items() if score > 0}
                    max_score = max(scores.values())
                    winning_labels = Set([label for label in scores if scores[label] == max_score])
                    most_voted_labels[k] = most_voted_labels[k].union(winning_labels)
                    self.logger.info('\tk = {} -> scores: {} -> winning: {}'.format(k, scores, most_voted_labels[k]))
                    if len(most_voted_labels[k]) == self.__training_set.num_classes():
                        break
        
        self.logger.info('\n\tdistances:')
        if self.logger.getEffectiveLevel() == 0:
            for distance, label in distances.get_items_ordered_so_far():
                self.logger.info('\t\t{}: {}'.format(label, distance))
        self.logger.info('\n')

        return most_voted_labels