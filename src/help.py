# -*- coding: utf-8 -*-
# =============================================================================
# File: help.py
# Updated: 05/11/2022
# =============================================================================
'''Tutorial script for using the tool'''
# =============================================================================
# Dependencies:
# -
# =============================================================================

from configparser import ConfigParser
from pick import pick
from sys import exit

settings_parser = ConfigParser()
settings_parser.read('settings.ini')

title = 'This is a guide to help you use kNAVe'

while True:
    options = [
        'Installation',
        'Usage',
        'Exit'
    ]

    option, _ = pick(options, title, indicator='=>')

    if option == 'Installation':
        text = title + (
            '\n\nTo install kNAVe:\n' +
            '\n  cd src' +
            '\n  pip install ./'
        )
        if pick(['Back', 'Exit'], text, indicator='=>')[0] == 'Exit':
            exit()

    elif option == 'Usage':
        text = title + (
            '\n\nTo run kNAVe:\n' +
            '\n  cd src' +
            '\n  python3 knave.py <config_file.ini>' +
            '\n  # or' + 
            '\n  python3 knave.py <config_file.ini> log' +
            '\n  # to obtain also a log file in \'{}\'\n'.format(settings_parser.get('DEFAULT', 'logs_dir')) +
            '\nNote: You can use \'...\' to match multiple files. For example:' +
            '\n  python3 knave.py ...    # match all files in \'{}\''.format(settings_parser.get('DEFAULT', 'config_dir')) +
            '\n  python3 knave.py str... # match all files in \'{}\' starting with \'str\''.format(settings_parser.get('DEFAULT', 'config_dir'))
        )
        while True:
            options = [
                'Configuration file',
                'Dataset',
                'Results',
                'Back',
                'Exit'
            ]
            option, _ = pick(options, text, indicator='=>')

            if option == 'Configuration file':
                text = title + (
                    '\n\n A configuration file \'config_file.ini\' must contain:\n' +
                    '\n  save_in = <path_to_results> # default = YYYY-MM-GG-hh-mm-ss' +
                    '\n  dataset_format = <value in {csv,libsvm,svmlight}> *' +
                    '\n  training_set = <training_set_name> *' +
                    '\n  test_set = <test_set_name> *' +
                    '\n  categorical_features = <integer_val_1> <integer_val_2> ... <integer_val_m>' + 
                    '\n  categories_<feature_number> = <literal_val_1> <literal_val_2> ... <literal_val_n>' +
                    '\n  feature_range = <min> <max>' + 
                    '\n  feature_range_<feature_number> = <min> <max>' + 
                    '\n  random = <value in {true,false}>' +
                    '\n  random_state = <integer_val> ' +
                    '\n  perturbation = <value in {hyper_rect,l_inf,noise_cat}> *' +
                    '\n  epsilons = <real_val_1> <real_val_2> ... <real_val_n>' +
                    '\n  epsilon = <real_val> * # only if perturbation = l_inf' +
                    '\n  noise = <real_val> * # only if perturbation = noise_cat' +
                    '\n  noise_type = <value in {hyper_rect,l_inf}>' +
                    '\n  cat_on = <integer_val_1> <integer_val_2> ... <integer_val_n>' +
                    '\n  k = <integer_val_1> <integer_val_2> ... <integer_val_n> *' +
                    '\n  distance_metric = <value in {euclidean,manhattan}> *' +
                    '\n  skip_ties = <value in {true,false}> # default = false \n' + 
                    '\n  * Required\n' +
                    '\n  Examples can be found in \'{}\'\n'.format(settings_parser.get('DEFAULT', 'config_dir')) +
                    '\n  Put your configuration files in \'{}\'\n'.format(settings_parser.get('DEFAULT', 'config_dir'))
                )
                if pick(['Back', 'Exit'], text, indicator='=>')[0] == 'Exit':
                    exit()
                break

            elif option == 'Dataset':
                text = title + (
                    '\n\nA dataset must be divided into training and test sets. If not do:\n' +
                    '\n  # Move your dataset to \'{}\''.format(settings_parser.get('DEFAULT', 'datasets_dir')) +
                    '\n  cd src' +
                    '\n  python3 split_dataset.py <dataset_name> <format in {csv,libsvm,svmlight}> <test_size %> [options]' +
                    '\n  # Training an test sets wll be saved in \'{}\'\n'.format(settings_parser.get('DEFAULT', 'datasets_dir')) +
                    '\n  [options]:' +
                    '\n    <random>' +
                    '\n    <random_state> # defualt = None\n' +
                    '\nExamples:' + 
                    '\n  python3 split_dataset.py german csv 20' +
                    '\n  python3 split_dataset.py australian libsvm 30' +
                    '\n  python3 split_dataset.py fourclass libsvm 35 random' +
                    '\n  python3 split_dataset.py diabetes svmlight 30 random 20\n' +
                    '\n\nDatasets can also be standardized:' +
                    '\n  # Move your dataset to \'{}\''.format(settings_parser.get('DEFAULT', 'datasets_dir')) +
                    '\n  cd src' +
                    '\n  python3 standardize_dataset.py <dataset_name> <format in {csv,libsvm,svmlight}>\n' +
                    '\nExamples:' + 
                    '\n  python3 standardize_dataset.py adult csv' +
                    '\n  python3 standardize_dataset.py letter libsvm'
                )
                if pick(['Back', 'Exit'], text, indicator='=>')[0] == 'Exit':
                    exit()
                break

            elif option == 'Results':
                text = title + (
                    '\n\nResults are saved in 3 files:\n' +
                    '\n  - details.csv: contains classifications for all processed feature vectors' +
                    '\n  - robustness.csv: contains robustness results' +
                    '\n  - stability.csv: contains stability results\n' +
                    '\n  Examples can be found in \'{}\'\n'.format(settings_parser.get('DEFAULT', 'results_dir'))
                )
                if pick(['Back', 'Exit'], text, indicator='=>')[0] == 'Exit':
                    exit()
                break

            elif option == 'Exit':
                exit() 
    else:
        exit()