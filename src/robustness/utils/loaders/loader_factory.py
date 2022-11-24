# -*- coding: utf-8 -*-
# =============================================================================
# File: loader_factory.py
# Updated: 05/11/2022
# =============================================================================
'''Create a loader'''
# =============================================================================
# Dependencies:
#   ./loader.py
#   ./csv_loader.py
#   ./libsvm_loader.py
#   ../../base.py
# =============================================================================

from .loader import Loader
from .csv_loader import CsvLoader
from .libsvm_loader import LibsvmLoader
from robustness import String

class LoaderFactory:
    '''Create a loader'''

    def create(self,
        dataset_format: String,
    ) -> Loader:
        '''
        Let the class initialize the object's attributes
        :param dataset_format: Format of files containing training and test sets
        :return: The best loader for the given dataset format
        '''
        if dataset_format == 'csv':
            return CsvLoader()
        elif dataset_format == 'libsvm' or dataset_format == 'svmlight':
            return LibsvmLoader()
        else:
            raise Exception('\'{}\' format not supported'.format(dataset_format))