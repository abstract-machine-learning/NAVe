# -*- coding: utf-8 -*-
# =============================================================================
# File: csv_loader.py
# Updated: 05/11/2022
# =============================================================================
'''Allow to load datasets in the csv format'''
# =============================================================================
# Dependencies:
#   ./loader.py
#   ../../base.py
# =============================================================================

from __future__ import annotations
from nptyping import NDArray
from numpy import array, hsplit
from os.path import join
from pandas import read_csv
from typing import Tuple

from .loader import Loader
from robustness.base import String

class CsvLoader(Loader):
    '''Load datasets in the csv format'''

    def load_from_file(self,
        training_set_name: String,
        test_set_name: String = ''
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        '''
        Load points and labels of the training and test sets
        :param training_set_name: Name and extension of the training set to load
        :param test_set_name: Name and extension of the test set to load (if available)
        :return: Points and labels of the training and test sets
        '''
        super().load_from_file(training_set_name, test_set_name)
        
        training_labels, training_points = hsplit(
            read_csv(
                join(self.get_datasets_dir_path(), training_set_name),
                header=None
            ).dropna().to_numpy(), [1]
        )
        training_labels = training_labels.flatten()

        if test_set_name == '' or test_set_name == training_set_name:
            return training_points, training_labels, array([]), array([])
            
        test_labels, test_points = hsplit(
            read_csv(
                join(self.get_datasets_dir_path(), test_set_name),
                header=None
            ).dropna().to_numpy(), [1]
        )
        test_labels = test_labels.flatten()
        
        return training_points, training_labels, test_points, test_labels