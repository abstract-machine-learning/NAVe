# -*- coding: utf-8 -*-
# =============================================================================
# File: abstract_classifier.py
# Updated: 05/11/2022
# =============================================================================
'''Define an abstract k-NN classifier'''
# =============================================================================
# Dependencies:
#   ../base.py
#   ../dataset.py
# =============================================================================

from abc import abstractmethod
from configparser import ConfigParser
from os import makedirs
from os.path import exists, join
from logging import basicConfig, getLogger
from typing import Type

from robustness import Dataset, Boolean, Integer, Literal, Map, Set, String, Vector
from robustness.abstract_domains import AbstractDomain

class AbstractClassifier:
    '''Represent an abstract k-NN classifier'''

    def __init__(self) -> None:
        '''
        Let the class initialize the object's attributes
        ''' 
        log_filename = '{}.log'.format(self.get_type())

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

    def set_log(self,
        on: Boolean
    ) -> None:
        '''
        Allow to activate or deactivate the logs
        :param on: Whether they should be activated or not
        '''
        return self.logger.setLevel(50 * (not on))

    def get_training_set(self) -> Dataset:
        '''
        Return the training set
        :return: Stored training set
        '''
        return self.__training_set

    def fit(self,
        training_set: Dataset
    ) -> None:
        '''
        Fit the classifier from the training dataset
        :param trainig_set: Training set to fit
        '''
        self.__training_set = training_set

    def get_type(self) -> String:
        '''
        Return the type of the abstract classifier
        :return: Class name
        '''
        return self.__class__.__name__

    @abstractmethod
    def classify(self,
        adv_region: Vector[Type[AbstractDomain]],
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
        pass