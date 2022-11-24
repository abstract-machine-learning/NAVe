# -*- coding: utf-8 -*-
# =============================================================================
# File: libsvm_loader.py
# Updated: 05/11/2022
# =============================================================================
'''Allow to load datasets in the svmlight / libsvm format'''
# =============================================================================
# Dependencies:
#   ./loader.py
#   ../error.py
#   ../../base.py
# =============================================================================

from __future__ import annotations
from nptyping import NDArray
from numpy import array
from os.path import join
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from typing import Tuple

from .loader import Loader
from robustness.base import String

class LibsvmLoader(Loader):
    '''Load datasets in the svmlight / libsvm format'''

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

        if test_set_name == '' or test_set_name == training_set_name:
            training_points, training_labels = load_svmlight_file(
                join(self.get_datasets_dir_path(), training_set_name)
            )
            training_points = training_points.toarray()

            return training_points, training_labels, array([]), array([])
        
        training_points, training_labels, test_points, test_labels = load_svmlight_files([
            join(self.get_datasets_dir_path(), training_set_name),
            join(self.get_datasets_dir_path(), test_set_name)
        ])
        training_points = training_points.toarray()
        test_points = test_points.toarray()

        return training_points, training_labels, test_points, test_labels