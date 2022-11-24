# -*- coding: utf-8 -*-
# =============================================================================
# File: split_dataset.py
# Updated: 05/11/2022
# =============================================================================
'''Allow to divide a dataset into training and test set'''
# =============================================================================
# Dependencies:
#   ./robustness/utils/error.py
#   ./robustness/utils/loaders/loader_factory.py
# =============================================================================

from configparser import ConfigParser
from os import makedirs
from os.path import exists, join
from math import ceil, floor
from numpy import array, concatenate
from pandas import DataFrame 
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
from sys import argv

from robustness import Integer, Real
from robustness.utils import Error
from robustness.utils.loaders import LoaderFactory

# argv[1] = training set name
# argv[2] = dataset format
# argv[3] = test set size
# argv[4] = 'random'
# argv[5] = random state

if __name__ == "__main__":
    if len(argv) < 4:
        Error('At least 4 arguments are expected')

    settings_parser = ConfigParser()
    settings_parser.read('settings.ini')
    if not exists(settings_parser.get('DEFAULT', 'config_dir')):
        makedirs(settings_parser.get('DEFAULT', 'config_dir'))

    datasets_dir_path = settings_parser.get('DEFAULT', 'datasets_dir')

    dataset_points, dataset_labels, _, _ = LoaderFactory().create(
        dataset_format=argv[2]
    ).load_from_file(training_set_name=argv[1])

    min_test_set_size = ceil(10000 / len(dataset_points)) / 100
    max_test_set_size = floor((len(dataset_points) - 1) * 10000 / len(dataset_points)) / 100
    try:
        if Real(argv[3]) < min_test_set_size or Real(argv[3]) > max_test_set_size:
            raise Exception('Invalid test set size')
    except:
        Error('Test set size must be a real number ranging in [{}, {}]'.format(min_test_set_size, max_test_set_size))
    if len(argv) > 4 and argv[4] != 'random':
        Error('Parameter \'{}\' not recognized. Expected value: \'random\''.format(argv[4]))
    if len(argv) > 5 and not argv[5].isdigit():
        Error('Random state \'{}\' not recognized. Expected value: positive integer'.format(argv[5]))

    training_points, test_points, training_labels, test_labels = train_test_split(
        dataset_points,
        dataset_labels,
        test_size=Integer(len(dataset_points) * (100 - Real(argv[3])) / 100),
        random_state= Integer(argv[5]) if len(argv) > 5 else None,
        shuffle=True if len(argv) > 5 else False
    )

    if argv[2] == 'csv':
        DataFrame(
            concatenate((array([training_labels]).T, training_points), axis=1)
        ).to_csv(join(datasets_dir_path, argv[1] + '_train.csv'), header=False, index=False)
        DataFrame(
            concatenate((array([test_labels]).T, test_points), axis=1)
        ).to_csv(join(datasets_dir_path, argv[1] + '_test.csv'), header=False, index=False)

    elif argv[2] == 'libsvm' or argv[2] == 'svmlight':
        dump_svmlight_file(training_points, training_labels, join(datasets_dir_path, argv[1] + '_train'))
        dump_svmlight_file(test_points, test_labels, join(datasets_dir_path, argv[1] + '_test'))
    
    else:
        Error('\'{}\' format not supported. Expected value: \'csv\', \'libsvm\' or \'svmlight\''.format(argv[2]))

    print('\nOperation completed successfully!')
    print('\n\'{}\' and \'{}\' have been saved in \'{}\'\n'.format(argv[1] + '_train', argv[1] + '_test', datasets_dir_path))