# -*- coding: utf-8 -*-
# =============================================================================
# File: knave.py
# Updated: 05/11/2022
# =============================================================================
'''Main program for the kNN abstract verifier'''
# =============================================================================
# Dependencies:
#   ./robustness/base.py
#   ./robustness/concrete_classifier.py
#   ./robustness/abstract_classifiers/interval_classifier.py
#   ./robustness/utils/error.py
#   ./robustness/utils/inizialize_main.py
# =============================================================================

from configparser import ConfigParser
from datetime import datetime
import glob
from os import listdir, makedirs
from os.path import exists, join
from pandas import DataFrame
from sys import argv
from time import time
from tqdm import tqdm
from typing import Any, Tuple

from robustness import ConcreteClassifier, Integer, Literal, Map, Real, Set, String, Vector
from robustness.abstract_classifiers import IntervalClassifier, RafClassifier
from robustness.utils import read_params, Error

import sys

write_log = False

def get_classification(
    test_point: Vector[Real],
    classifier: ConcreteClassifier | IntervalClassifier | RafClassifier,
    params: Map[String, Any],
) -> Tuple[Map[Integer, Set[Literal]], Real]:
    '''
    Provide the classification of a point and its execution time
    :param test_point: Target test point
    :param classifier: Classifier to use
    :param params: Input params
    :return: Point classification and execution time
    '''
    start_time = time()

    most_voted_labels = {k: Set() for k in params['k_values']}
    k_values_to_use = [k for k in params['k_values']]
    
    for _ in range(params['perturbation'].num_adv_regions()):
        adv_region = params['perturbation'].perturb(test_point)
        for k, labels in classifier.classify(adv_region, k_values_to_use, params['distance_metric']).items():
            most_voted_labels[k] = most_voted_labels[k].union(labels)
            if len(most_voted_labels[k]) == params['training_set'].num_classes():
                k_values_to_use.remove(k)
        if len(k_values_to_use) == 0:
            break

    return most_voted_labels, time() - start_time

def perform_concrete_classification(
    params: Map[String, Any],
    results_dir_path: String
) -> None:
    '''
    Main program for concrete classification
    :param params: Input params
    :param results_dir_path: Directory for the results
    '''
    details = []
    stable_yes_cnt = []
    stable_no_cnt = []
    robust_yes_cnt = []
    robust_no_cnt = []
    runtime = 0

    for _ in params['k_values']:
        details.append(
            DataFrame({
                'Robust': [],
                'Stable': [],
                'Classification': []
            })
        )
        stable_yes_cnt.append(0)
        stable_no_cnt.append(0)
        robust_yes_cnt.append(0)
        robust_no_cnt.append(0)

    concrete_classifier = ConcreteClassifier()
    concrete_classifier.fit(params['training_set'])
    concrete_classifier.set_log(write_log)

    classified_points = 0

    progress_bar = tqdm(
        zip(params['test_set'].get_points(), params['test_set'].get_labels()),
        total=params['num_test'],
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]',
        desc='Verifying',
        postfix='ROB=?%, STAB=?%'
    )

    for i, (test_point, test_label) in enumerate(progress_bar):
        if classified_points == params['num_test']:
            break

        if params['skip_ties']:
            for labels in concrete_classifier.classify(test_point, params['k_values'], params['distance_metric']).values():
                if len(labels) > 1:
                    break
            else:
                break
            continue

        most_voted_labels, exec_time = get_classification(test_point, concrete_classifier, params)
        runtime += exec_time
        classified_points += 1
        
        for j, (k, labels) in enumerate(most_voted_labels.items()):
            if len(labels) == 1:
                is_stable = 'Yes'
                stable_yes_cnt[j] += 1
                if test_label in labels:
                    is_robust = 'Yes' 
                    robust_yes_cnt[j] += 1
                else:
                    is_robust = 'No'
                    robust_no_cnt[j] += 1
            else:
                is_stable = 'No'
                stable_no_cnt[j] += 1
                is_robust = 'No'
                robust_no_cnt[j] += 1

            details[j].loc[len(details[j].index)] = [is_robust, is_stable, labels]

        progress_bar.set_postfix_str('ROB={}%, STAB={}%'.format(
            round(sum(robust_yes_cnt) / (classified_points * len(params['k_values'])) * 100, 1),
            round(sum(stable_yes_cnt) / (classified_points * len(params['k_values'])) * 100, 1)
        ))

        if classified_points == params['num_test'] or i + 1 == params['test_set'].num_points():
            progress_bar.set_description('Completed')

    for i, k in enumerate(params['k_values']):
        if not exists(join(results_dir_path, 'k{}'.format(k))):
            makedirs(join(results_dir_path, 'k{}'.format(k)))

        details[i].to_csv(join(results_dir_path, 'k{}'.format(k), 'details.csv'))

        robustness = DataFrame({
            '# Yes' : [robust_yes_cnt[i]],
            '# No': [robust_no_cnt[i]],
            'Robustness': ['{}%'.format(
                round(robust_yes_cnt[i] / classified_points * 100, 1)
                    if classified_points > 0 else '-'
            )]
        })
        robustness.to_csv(join(results_dir_path, 'k{}'.format(k), 'robustness.csv'))

        stability = DataFrame({
            '# Yes' : [stable_yes_cnt[i]],
            '# No': [stable_no_cnt[i]],
            'Stability': ['{}%'.format(round(
                stable_yes_cnt[i] / classified_points * 100, 1)
                    if classified_points > 0 else '-'
            )]
        })
        stability.to_csv(join(results_dir_path, 'k{}'.format(k), 'stability.csv'))

    with open(join(results_dir_path, 'runtime.txt'), 'w') as file:
        file.write('{} seconds'.format(round(runtime)))

    print('\nThe results have been saved in \'{}\'\n'.format(results_dir_path))

def perform_abstract_classification(
    params: Map[String, Any],
    results_dir_path: String
) -> None:
    '''
    Main program for abstract classification
    :param params: Input params
    :param results_dir_path: Directory for the results
    '''
    details = []
    stable_yes_cnt = []
    stable_do_not_know_cnt = []
    robust_yes_cnt = []
    robust_no_cnt = []
    robust_do_not_know_cnt = []
    runtime = 0

    for _ in params['k_values']:
        details.append(
            DataFrame({
                'Robust': [],
                'Stable': [],
                'Classification': []
            })
        )
        stable_yes_cnt.append(0)
        stable_do_not_know_cnt.append(0)
        robust_yes_cnt.append(0)
        robust_no_cnt.append(0)
        robust_do_not_know_cnt.append(0)

    abstract_classifier = IntervalClassifier()
    if params['abstraction'] == 'raf':
        abstract_classifier = RafClassifier()
    abstract_classifier.fit(params['training_set'])
    abstract_classifier.set_log(write_log)

    if params['skip_ties']:
        concrete_classifier = ConcreteClassifier()
        concrete_classifier.fit(params['training_set'])

    classified_points = 0

    progress_bar = tqdm(
        zip(params['test_set'].get_points(), params['test_set'].get_labels()),
        total=params['num_test'],
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]',
        desc='Verifying',
        postfix='ROB=?%, STAB=?%'
    )

    for i, (test_point, test_label) in enumerate(progress_bar):
        if classified_points == params['num_test']:
            break

        if params['skip_ties']:
            for labels in concrete_classifier.classify(test_point, params['k_values'], params['distance_metric']).values():
                if len(labels) > 1:
                    break
            else:
                break
            continue

        most_voted_labels, exec_time = get_classification(test_point, abstract_classifier, params)
        runtime += exec_time
        classified_points += 1

        for j, (k, labels) in enumerate(most_voted_labels.items()):
            if len(labels) == 1:
                is_stable = 'Yes'
                stable_yes_cnt[j] += 1
                if test_label in labels:
                    is_robust = 'Yes'
                    robust_yes_cnt[j] += 1
                else:
                    is_robust = 'No'
                    robust_no_cnt[j] += 1
            else:
                is_stable = 'Do not know'
                stable_do_not_know_cnt[j] += 1
                if test_label in labels:
                    is_robust = 'Do not know' 
                    robust_do_not_know_cnt[j] += 1
                else:
                    is_robust = 'No'
                    robust_no_cnt[j] += 1

            details[j].loc[len(details[j].index)] = [is_robust, is_stable, labels]

        progress_bar.set_postfix_str('ROB={}%, STAB={}%'.format(
            round(sum(robust_yes_cnt) / (classified_points * len(params['k_values'])) * 100, 1),
            round(sum(stable_yes_cnt) / (classified_points * len(params['k_values'])) * 100, 1)
        ))

        if classified_points == params['num_test'] or i + 1 == params['test_set'].num_points():
            progress_bar.set_description('Completed')

    for i, k in enumerate(params['k_values']):
        if not exists(join(results_dir_path, 'k{}'.format(k))):
            makedirs(join(results_dir_path, 'k{}'.format(k)))

        details[i].to_csv(join(results_dir_path, 'k{}'.format(k), 'details.csv'))

        robustness = DataFrame({
            '# Yes' : [robust_yes_cnt[i]],
            '# No': [robust_no_cnt[i]],
            '# Do not know': [robust_do_not_know_cnt[i]],
            'Proved Robustness': ['{}%'.format(
                round(robust_yes_cnt[i] / classified_points * 100, 1)
                    if classified_points > 0 else '-'
            )]
        })
        robustness.to_csv(join(results_dir_path, 'k{}'.format(k), 'robustness.csv'))

        stability = DataFrame({
            '# Yes' : [stable_yes_cnt[i]],
            '# Do not know': [stable_do_not_know_cnt[i]],
            'Proved Stability': ['{}%'.format(round(
                stable_yes_cnt[i] / classified_points * 100, 1)
                    if classified_points > 0 else '-'
            )]
        })
        stability.to_csv(join(results_dir_path, 'k{}'.format(k), 'stability.csv'))

    with open(join(results_dir_path, 'runtime.txt'), 'w') as file:
        file.write('{} seconds'.format(round(runtime)))

    print('\nThe results have been saved in \'{}\'\n'.format(results_dir_path))

def main(input_file_path: String) -> None:
    '''
    Main program
    :param input_file_path: Path of the file to import
    '''
    params = read_params(input_file_path)

    results_dir_path = join(join(
        settings_parser.get('DEFAULT', 'results_dir'),
        params['save_in'] if params['save_in'] != '' else '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now())
    ), params['distance_metric'], params['abstraction'])

    if not exists(results_dir_path):
        makedirs(results_dir_path)

    with open(join(results_dir_path, 'config.ini'), 'w') as backup_input_file:
        config_parser = ConfigParser()
        config_parser.read(input_file_path)
        config_parser.write(backup_input_file)

    if params['classifier'] == 'concrete':
        perform_concrete_classification(params, results_dir_path)
    else:
        perform_abstract_classification(params, results_dir_path)


# argv[1] = configuration file name
# argv[2] = 'log'

if __name__ == "__main__":
    if len(argv) < 2:
        Error('Missing input file name')
    if len(argv) > 2:
        if argv[2] == 'log':
            write_log = True
        else:
            Error('Parameter \'{}\' not recognized. Expected value: \'log\''.format(argv[2]))
    else:
        write_log = False

    settings_parser = ConfigParser()
    settings_parser.read('settings.ini')

    if not exists(settings_parser.get('DEFAULT', 'config_dir')):
        makedirs(settings_parser.get('DEFAULT', 'config_dir'))

    if '...' in argv[1]:
        filename_init = argv[1][:argv[1].index('...')]
        files = []
        path = settings_parser.get('DEFAULT', 'config_dir') + '/' + filename_init + '*.ini'
        for file in glob.glob(path):
            files.append(file)

        if len(files) == 0:
            Error('No files starting with \'{}\' were found in \'{}\''.format(argv[1][:-3], settings_parser.get('DEFAULT', 'config_dir')))

        for i, filename in enumerate(files):
            print('\nFile \'{}\' [{}/{}]'.format(filename, i + 1, len(files)))
            main(filename)

    else:
        if exists(join(settings_parser.get('DEFAULT', 'config_dir'), argv[1])):
            input_file_path = join(settings_parser.get('DEFAULT', 'config_dir'), argv[1])
            print('\nFile \'{}\''.format(argv[1]))
            main(input_file_path)

        else:
            Error('File \'{}\' does not exist in \'{}\''.format(argv[1], settings_parser.get('DEFAULT', 'config_dir')))