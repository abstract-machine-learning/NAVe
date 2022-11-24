# -*- coding: utf-8 -*-
# =============================================================================
# File: standardize_dataset.py
# Updated: 05/11/2022
# =============================================================================
'''Allow to standardize the numerical features of a dataset'''
# =============================================================================
# Dependencies:
#   ./robustness/utils/error.py
#   ./robustness/utils/loaders/loader_factory.py
# =============================================================================

from configparser import ConfigParser
from os import makedirs
from os.path import exists, join
from numpy import array, concatenate
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import StandardScaler
from sys import argv

from robustness import String
from robustness.utils import Error
from robustness.utils.loaders import LoaderFactory

# argv[1] = training set name
# argv[2] = dataset format

if __name__ == "__main__":
    if len(argv) < 3:
        Error('At least 3 arguments are expected')

    settings_parser = ConfigParser()
    settings_parser.read('settings.ini')
    if not exists(settings_parser.get('DEFAULT', 'config_dir')):
        makedirs(settings_parser.get('DEFAULT', 'config_dir'))

    datasets_dir_path = settings_parser.get('DEFAULT', 'datasets_dir')

    dataset_points, dataset_labels, _, _ = LoaderFactory().create(
        dataset_format=argv[2]
    ).load_from_file(training_set_name=argv[1])

    numerical_indexes = []
    for i, feature in enumerate(dataset_points[0]):
        if not isinstance(feature, String):
            numerical_indexes.append(i)

    standardizer = ColumnTransformer(
        [('', StandardScaler(), numerical_indexes)]
    )
    for i, standardized_features in enumerate(standardizer.fit_transform(dataset_points)):
        for feature_index, feature in zip(numerical_indexes, standardized_features):
            dataset_points[i][feature_index] = feature

    if argv[2] == 'csv':
        DataFrame(
            concatenate((array([dataset_labels]).T, dataset_points), axis=1)
        ).to_csv(join(datasets_dir_path, argv[1] + '_standardized.csv'), header=False, index=False)

    elif argv[2] == 'libsvm' or argv[2] == 'svmlight':
        dump_svmlight_file(dataset_points, dataset_labels, join(datasets_dir_path, argv[1] + '_standardized'))
    
    else:
        Error('\'{}\' format not supported. Expected value: \'csv\', \'libsvm\' or \'svmlight\''.format(argv[2]))

    print('\nStandardization completed successfully!')
    print('\n\'{}\' has been saved in \'{}\'\n'.format(argv[1] + '_standardized', datasets_dir_path))