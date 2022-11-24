# -*- coding: utf-8 -*-
# =============================================================================
# File: loader.py
# Updated: 05/11/2022
# =============================================================================
'''Allow to load different dataset formats'''
# =============================================================================
# Dependencies:
#   ../error.py
#   ../preprocessing.py
#   ../../base.py
#   ../../dataset.py
# =============================================================================

from __future__ import annotations
from abc import abstractmethod
from configparser import ConfigParser
from nptyping import NDArray
from os import makedirs
from os.path import exists, join
from sklearn.utils import shuffle
from typing import Tuple

from ..error import Error
from ..preprocessing import scale_features
from robustness import Dataset, Boolean, Integer, Literal, Map, Real, Set, String, Vector

class Loader:
    '''Load different dataset formats'''

    def __init__(self) -> None:
        '''
        Let the class initialize the object's attributes
        '''
        settings_parser = ConfigParser()
        settings_parser.read('settings.ini')

        if not exists(settings_parser.get('DEFAULT', 'datasets_dir')):
            makedirs(settings_parser.get('DEFAULT', 'datasets_dir'))

        self.__datasets_dir_path = settings_parser.get('DEFAULT', 'datasets_dir')

    def get_datasets_dir_path(self) -> String:
        '''
        Return the path to the datasets directory
        :return: The path to the datasets directory
        '''
        return self.__datasets_dir_path

    def load(self,
        training_set_name: String,
        test_set_name: String,
        random: Boolean = True,
        random_state: Integer | None = None,
        feature_range: Map[Literal, Tuple[Real, Real]] = {}, 
        categorical_indexes: Vector[Integer] = [],
        categories_list: Vector[Vector[Literal]] = [],
    ) -> Tuple[Dataset, Dataset]:
        '''
        Load training and test sets
        :param training_set_name: Name and extension of the training set to load
        :param test_set_name: Name and extension of the test set to load
        :param random: Whether or not to randomize the selection of the test points
        :param random_state: Random number generation for shuffling the data
        :param feature_range: Minimum and maximum value of the numerical features of the points
        :param categorical_indexes: Indexes of the categorical features
        :param categories_list: Holds the categories expected in the every categorical feature
        :return: Trainig and test sets
        '''
        training_points, training_labels, test_points, test_labels = self.load_from_file(
            training_set_name,
            test_set_name
        )

        if len(training_points) == 0:
            Error('Training set \'{}\' is empty'.format(training_set_name))
        if len(test_points) == 0:
            Error('Test set \'{}\' is empty'.format(test_set_name))
        if training_points.shape[1] != test_points.shape[1]:
            Error('Training set \'{}\' and test set \'{}\' have a different number of features'.format(training_set_name, test_set_name))

        for feature_index, categories in zip(categorical_indexes, categories_list):
            if len(categories) == 0 and feature_index < training_points.shape[1]:
                new_categories = Set([point[feature_index] for point in training_points])
                new_categories = new_categories.union(Set([point[feature_index] for point in test_points]))
                categories.extend([*new_categories])

        if random:
            test_points, test_labels = shuffle(test_points, test_labels, random_state)

        training_points, test_points = scale_features(
            training_points,
            test_points,
            feature_range,
            categorical_indexes,
            categories_list,
        )

        training_points = training_points.tolist()
        training_labels = training_labels.tolist()
        test_points = test_points.tolist()
        test_labels = test_labels.tolist()

        return Dataset(training_points, training_labels), Dataset(test_points, test_labels)

    def load_from_file(self,
        training_set_name: String,
        test_set_name: String = '',
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        '''
        Load points and labels of the training and test sets
        :param training_set_name: Name and extension of the training set to load
        :param test_set_name: Name and extension of the test set to load (if available)
        :return: Points and labels of the training and test sets
        '''
        if not exists(join(self.get_datasets_dir_path(), training_set_name)):
            Error('There is no dataset \'{}\' in \'{}\''.format(training_set_name, self.get_datasets_dir_path()))
        
        if test_set_name != '' and not exists(join(self.get_datasets_dir_path(), test_set_name)):
            Error('There is no dataset \'{}\' in \'{}\''.format(test_set_name, self.get_datasets_dir_path()))