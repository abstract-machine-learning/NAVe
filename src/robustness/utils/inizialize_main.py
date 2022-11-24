# -*- coding: utf-8 -*-
# =============================================================================
# File: inizialize_main.py
# Updated: 05/11/2022
# =============================================================================
'''Contain functions for reading input data'''
# =============================================================================
# Dependencies:
#   ./error.py
#   ./loaders/loader_factory.py
#   ../base.py
#   ../perturbations/hyperrectangle.py
#   ../perturbations/l_infinity.py
#   ../perturbations/noise_cat.py
# =============================================================================

from configparser import ConfigParser
from typing import Type

from robustness import Dataset, Boolean, Number, Integer, Literal, Map, Real, Set, String, Vector
from robustness.perturbations import Hyperrectangle, Linfinity, NoiseCat
from robustness.utils import Error
from robustness.utils.loaders import LoaderFactory

def read_params(input_file_path: String) -> Map[String, Dataset | Integer | String | Vector[Number]]:
    '''
    Read the data contained in the input file
    :param input_file_path: Path of the file to import
    :return: All relevant data contained in the input file
    '''
    config_parser = ConfigParser()
    config_parser.read(input_file_path)

    def get_string(
        key: String,
        required: Boolean = True
    ) -> String:
        if required:
            if not config_parser.has_option('DEFAULT', key):
                Error('Attribute \'{}\' in \'{}\' is missing'.format(key, input_file_path))
            elif config_parser.get('DEFAULT', key) == '':
                Error('Attribute \'{}\' in \'{}\' cannot be empty'.format(key, input_file_path))
        return config_parser.get('DEFAULT', key) if config_parser.has_option('DEFAULT', key) else ''

    def get_boolean(
        key: String,
        empty_is_true: Boolean = False,
        required: Boolean = True
    ) -> Boolean:
        boolean_as_string = get_string(key, required)
        if boolean_as_string != 'true' and boolean_as_string != 'false' and boolean_as_string != '':
            Error('Attribute \'{}\' in \'{}\' can only be \'true\', \'false\' or empty'.format(key, input_file_path))
        return boolean_as_string != 'false' if empty_is_true else boolean_as_string == 'true'

    def get_pos_integer(
        key: String,
        required: Boolean = True
    ) -> Integer | None:
        if get_string(key, required) == '':
            return None
        if get_string(key, required).isdigit() and Integer(get_string(key, required)) > 0:
            return Integer(get_string(key, required))
        Error('Attribute \'{}\' in \'{}\' can only be a positive integer other than 0'.format(key, input_file_path))

    def get_real(
        key: String,
        required: Boolean = True
    ) -> Integer | None:
        if get_string(key, required) == '':
            return None
        try:
            return Real(get_string(key, required))
        except:
            Error('Attribute \'{}\' in \'{}\' can only be a real number'.format(key, input_file_path))

    def get_vector(
        key: String,
        type: Type,
        add: Number = 0,
        required: Boolean = True
    ) -> Vector[Integer | Real | String | Literal]:
        vector = []
        if type == Integer: # (positive integers other than 0)
            for value in get_string(key, required).split():
                if not value.isdigit() or Integer(value) <= 0:
                    Error('Attribute \'{}\' in \'{}\' can only contain positive integers other than 0'.format(key, input_file_path))
                vector.append(Integer(value) + add)
        if type == Real:
            for value in get_string(key, required).split():
                try:
                    vector.append(Real(value) + add)
                except:
                    Error('Attribute \'{}\' in \'{}\' can only contain real numbers'.format(key, input_file_path))
        if type == String:
            vector = [value for value in get_string(key, required).split()]
        if type == Literal:
            vector = [Integer(value) + add if value.isdigit() else value for value in get_string(key, required).split()]
        return vector

    #-----------------------------------------------------------------------------------------------------------------
    # categorical features

    categorical_indexes = sorted(
        [*Set(get_vector('categorical_features', type=Integer, add=-1, required=False))]
    )

    categories_list = []
    for feature_index in categorical_indexes:
        categories_list.append(
            get_vector('categories_{}'.format(feature_index + 1), type=Literal, required=False)
        )

    #-----------------------------------------------------------------------------------------------------------------
    # feature range

    feature_range = {'all': tuple(get_vector('feature_range', type=Real, required=False))}
    if len(feature_range['all']) == 0:
        del feature_range['all']
    if len(feature_range) != 0 and len(feature_range['all']) != 2:
        Error('Attribute \'feature_range\' in \'{}\' can only contain two real numbers (min and max)'.format(input_file_path))

    if len(feature_range) == 0:
        for name, _ in config_parser.items('DEFAULT'):
            if name.startswith('feature_range_'):
                num_feature_index = Integer(name[14:]) - 1
                if num_feature_index not in categorical_indexes:
                    Error('Attribute \'{}\' in \'{}\' refers to a categorical feature instead of a numerical one'.format(name, input_file_path))
                before_cnt = 0
                for cat_feature_index in categorical_indexes:
                    if cat_feature_index > num_feature_index:
                        break
                    before_cnt += 1
                num_feature_index -= before_cnt
                feature_range[num_feature_index] = tuple(get_vector(key=name, type=Real, required=False))
                if len(feature_range[num_feature_index]) != 2:
                    Error('Attribute \'{}\' in \'{}\' can only contain two real numbers (min and max)'.format(name, input_file_path))

    #-----------------------------------------------------------------------------------------------------------------
    # training and test sets

    dataset_format = get_string('dataset_format', required=True)
    if dataset_format != 'csv' and dataset_format != 'libsvm' and dataset_format != 'svmlight':
        Error('Attribute \'dataset_format\' in \'{}\' can only be \'csv\', \'libsvm\' or \'svmlight\''.format(input_file_path))

    training_set, test_set = LoaderFactory().create(
        get_string('dataset_format', required=True)
    ).load(
        get_string('training_set', required=True),
        get_string('test_set', required=True),
        get_boolean('random', required=True),
        get_pos_integer('random_state', required=False),
        feature_range,
        categorical_indexes,
        categories_list
    )

    for feature_index in categorical_indexes:
        if feature_index >= training_set.num_features():
            Error('Attribute \'categorical_features\' in \'{}\' can only contain positive integers ranging in [1, {}]'.format(input_file_path, training_set.num_features()))

    print('\nTraining set')
    training_set.print_info()
    print('\nTest set')
    test_set.print_info()

    #-----------------------------------------------------------------------------------------------------------------
    # number of test points

    num_test = get_pos_integer('num_test', required=False)
    if num_test == None:
        num_test = test_set.num_points()
    else:
        num_test = min(num_test, test_set.num_points())

    #-----------------------------------------------------------------------------------------------------------------
    # perturbation and classifier

    perturbation_name = get_string('perturbation', required=True)
    start_perturbation_from = sum(len(categories) if len(categories) > 2 else 1 for categories in categories_list)
    numerical_features_size = training_set.num_features() - start_perturbation_from

    if 'all' in feature_range:
        feature_range = [(0.0, 1.0) for _ in range(numerical_features_size)]
    else:
        feature_range = [
            (0.0, 1.0) if feature_index in feature_range else None for feature_index in range(numerical_features_size)
        ]

    classifier = 'concrete'

    if perturbation_name == 'hyper_rect':
        epsilons = get_vector('epsilons', type=Real)
        if len(epsilons) != numerical_features_size:
            Error('Attribute \'epsilons\' in \'{}\' must contain {} real numbers, one for each feature'.format(input_file_path, numerical_features_size))

        for epsilon in epsilons:
            if epsilon < 0 or epsilon > 1.0:
                Error('Attribute \'epsilons\' in \'{}\' can only contain real numbers ranging in [0, 1]'.format(input_file_path))
            if epsilon > 0.0:
                classifier = 'abstract'

        perturbation = Hyperrectangle(
            epsilons,
            feature_range,
            start_perturbation_from
        )
        print('\nPerturbation:\t {}'.format(perturbation_name))
        print('Epsilons:\t {}\n'.format(epsilons))

    elif perturbation_name == 'l_inf':
        epsilon = get_real('epsilon')
        if epsilon < 0 or epsilon > 1.0:
            Error('Attribute \'epsilon\' in \'{}\' can only be a real number ranging in [0, 1]'.format(input_file_path))

        if epsilon > 0.0:
            classifier = 'abstract'

        perturbation = Linfinity(
            epsilon,
            feature_range,
            start_perturbation_from
        )
        print('\nPerturbation:\t {}'.format(perturbation_name))
        print('Epsilon:\t {}\n'.format(epsilon))

    elif perturbation_name == 'noise-cat':
        noise_type_name = get_string('noise_type', required=True)

        if noise_type_name == 'hyper_rect':
            epsilons = get_vector('epsilons', type=Real)
            if len(epsilons) != numerical_features_size:
                Error('Attribute \'epsilons\' in \'{}\' must contain {} real numbers, one for each feature'.format(input_file_path, numerical_features_size))
            for epsilon in epsilons:
                if epsilon < 0 or epsilon > 1.0:
                    Error('Attribute \'epsilon\' in \'{}\' can only be a real number ranging in [0, 1]'.format(input_file_path))
                if epsilon > 0.0:
                    classifier = 'abstract'

            noise = Hyperrectangle(
                epsilons,
                feature_range,
                start_perturbation_from
            )

        elif noise_type_name == 'l_inf':
            epsilon = get_real('epsilon')
            if epsilon < 0 or epsilon > 1.0:
                Error('Attribute \'epsilon\' in \'{}\' can only be a real number ranging in [0, 1]'.format(input_file_path))
            if epsilon > 0.0:
                classifier = 'abstract'

            noise = Linfinity(
                epsilon,
                feature_range,
                start_perturbation_from
            )

        else:
            Error('Attribute \'noise_type\' in \'{}\' can only be \'hyper_rect\' or \'l_inf\''.format(input_file_path))

        cat_indexes = []
        for feature_index in get_vector('cat_on', type=Integer, add=-1, required=True):
            if not feature_index in categorical_indexes:
                 Error('Attribute \'noise_type\' in \'{}\' can only be \'hyper_rect\' or \'l_inf\''.format(input_file_path))

            categories_index = categorical_indexes.index(feature_index)
            if len(categories_list[categories_index]) <= 2:
                from_index = sum(
                    1 for categories in categories_list[:categories_index] if len(categories) <= 2
                )
                num_values = 2
            else:
                from_index = sum(1 for categories in categories_list if len(categories) <= 2) + sum(
                    len(categories) for categories in categories_list[:categories_index] if len(categories) > 2
                )
                num_values = len(categories_list[categories_index])

            cat_indexes.append((from_index, num_values))

        perturbation = NoiseCat(
            noise,
            cat_indexes
        )
        print('\nPerturbation:\t NOISE-CAT')
        print('NOISE type:\t {}'.format(noise_type_name))
        print('NOISE:\t\t {}'.format(epsilon))
        print('CAT on:\t\t {}\n'.format(get_vector('cat_on', type=Integer, required=True)))

    else:
        Error('Attribute \'perturbation\' in \'{}\' can only be \'hyper_rect\', \'l_inf\' or \'noise-cat\''.format(input_file_path))

    #-----------------------------------------------------------------------------------------------------------------
    # values of 'k'

    k_values = get_vector('k', type=Integer, required=True)
    for k in k_values:
        if k <= 0 or k > training_set.num_points():
            Error('Attribute \'k\' in \'{}\' can only be an integer ranging in [1, {}]'.format(input_file_path, training_set.num_points()))

    #-----------------------------------------------------------------------------------------------------------------
    # distance metric

    distance_metric = get_string('distance_metric', required=True)
    if distance_metric != 'manhattan' and distance_metric != 'euclidean':
        Error('Attribute \'distance_metric\' in \'{}\' can only be \'manhattan\' or \'euclidean\''.format(input_file_path))

    return {
        'classifier': classifier,
        'training_set': training_set,
        'test_set': test_set,
        'num_test': num_test,
        'perturbation': perturbation,
        'k_values': k_values,
        'distance_metric': distance_metric,
        'skip_ties': get_boolean('skip_ties', empty_is_true=False, required=False),
        'save_in': get_string('save_in', required=False)
    }