# -*- coding: utf-8 -*-
# =============================================================================
# File: raf_classifier.py
# Updated: 05/11/2022
# =============================================================================
'''Implementation of the raf k-NN classifier'''
# =============================================================================
# Dependencies:
#   ./abstract_classifier.py
#   ../base.py
#   ../dataset.py
#   ../abstract_domains/raf.py
#   ../utils/distances.py
#   ../utils/min_heap.py
# =============================================================================

from __future__ import annotations
from math import ceil
from typing import Tuple, Type

from .abstract_classifier import AbstractClassifier
from robustness import Boolean, Integer, Literal, Map, Set, String, Vector
from robustness.abstract_domains import Interval, Raf
from robustness.utils import compute_distance, MinHeap

class RafClassifier(AbstractClassifier):
    def __compute_abstract_distances(self,
        adv_region: Vector[Type[Raf]],
        distance_metric: String
    ) -> MinHeap:
        distances = []
        for train_point, train_label in zip(self.get_training_set().get_points(), self.get_training_set().get_labels()):
            distances.append((
                compute_distance(adv_region, train_point, distance_metric, init=Raf(center=0, noise=0, linear=None, dimensions=len(train_point))),
                train_label
            ))
        return MinHeap(distances)

    def __get_bounds_for_labels(self,
        distances: MinHeap,
        k: Integer
    ) -> Map[Literal, Type[Raf]]:
        '''
        Returns of occurrence for the labels
        :param distances: Abstract distances of each point in the training set to the adversarial region
        :param k: Number of neighbors to consider
        :return: Bounds of occurrence for the labels
        '''
        def certainly_considered() -> Boolean:
            for j in range (k + 1, distances.get_size() + 1):
                jth_distance, jth_label = distances.get_nth_smallest(j)
                if not ith_distance.strictly_dominated_by(jth_distance):
                    if ith_label != jth_label:
                        return False
                else:
                    break
            return True
        
        def possibly_considered() -> Tuple[Boolean, Boolean]:
            keep_searching = False
            for i in range (k, 0, -1):
                ith_distance, ith_label = distances.get_nth_smallest(i)
                if not jth_distance.strictly_dominates(ith_distance):
                    keep_searching = True
                    if jth_label != ith_label:
                        return True, keep_searching
            return False, keep_searching

        bounds = {label : Interval(0, 0) for label in self.get_training_set().get_classes()}
        certainly_closer = 0

        for i in range(1, k + 1):
            ith_distance, ith_label = distances.get_nth_smallest(i)
            bounds[ith_label].ub += 1
            if certainly_considered():
                bounds[ith_label].lb += 1
                certainly_closer += 1

        uncertainty = k - certainly_closer
        
        if uncertainty > 0:
            for j in range(k + 1, distances.get_size() + 1):
                jth_distance, jth_label = distances.get_nth_smallest(j)
                possibly_closer, keep_searching = possibly_considered()
                if possibly_closer:
                    if bounds[jth_label].ub - bounds[jth_label].lb < uncertainty:
                        bounds[jth_label].ub += 1
                elif not keep_searching:
                    break
        
        return bounds

    def __get_most_voted_labels(self,
        bounds: Map[Literal, Type[Raf]],
        k: Integer
    ) -> Set[Literal]:
        '''
        Return the most voted labels for a given k
        :param bounds: Bounds of occurrence for the labels
        :param k: Number of neighbors to consider
        :return: The most voted labels for a single k
        '''
        for label in [*bounds]:
            if bounds[label].dominated_by(0):
                del bounds[label]

        if len(bounds) == 1 or k == 1:
            return Set(bounds.keys())
        
        sum_ubs = sum(bounds[label].ub for label in bounds)
        for label in bounds:
            bounds[label].lb = max(bounds[label].lb, k - sum_ubs + bounds[label].ub)

        min_score_to_win = ceil(k / len(bounds))
        
        winning_labels = Set()
        for ith_label in bounds:
            if bounds[ith_label].strictly_dominated_by(min_score_to_win):
                continue
            for jth_label in bounds:
                if bounds[ith_label].strictly_dominated_by(bounds[jth_label]):
                    break
            else:
                winning_labels.add(ith_label)
    
        return winning_labels

    def classify(self,
        adv_region: Vector[Type[Raf]],
        k_values: Vector[Integer],
        distance_metric: String = 'euclidean'
    ) -> Map[Integer, Set[Literal]]:
        '''
        Perform the abstract classification
        :param adv_region: Target adversarial region
        :param k_values: Number of neighbors to consider (one or more values)
        :param distance_metric: Metric to evaluate the abstract distance between two points
        :return: The most voted labels for each k
        '''        
        self.logger.info('- adversarial region: {}\n'.format(adv_region))
        
        raf_adv_region = []
        for i in range(0, len(adv_region)):
            box = adv_region[i]
            c = 0.5 * (box.lb + box.ub)
            a = 0.5 * (box.ub - box.lb)
            raf = Raf(c, None, 0, len(adv_region))
            raf.linear[i] = a
            raf_adv_region.append(raf)
        distances = self.__compute_abstract_distances(raf_adv_region, distance_metric)

        most_voted_labels = {}
        for k in k_values:
            bounds = self.__get_bounds_for_labels(distances, k)
            most_voted_labels[k] = self.__get_most_voted_labels(bounds, k)
            self.logger.info('\tk = {} -> bounds: {} -> winning: {}'.format(k, bounds, most_voted_labels[k]))

        self.logger.info('\n\tdistances:')
        if self.logger.getEffectiveLevel() == 0:
            for distance, label in distances.get_items_ordered_so_far():
                self.logger.info('\t\t{}: {}'.format(label, distance))
        self.logger.info('\n')

        return most_voted_labels