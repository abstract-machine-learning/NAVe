# -*- coding: utf-8 -*-
# =============================================================================
# File: preprocessing.py
# Updated: 05/11/2022
# =============================================================================
'''Contain functions for data preprocessing'''
# =============================================================================
# Dependencies:
#   ../base.py
# =============================================================================

from nptyping import NDArray
from numpy import array, concatenate, delete, hsplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from typing import Tuple

from robustness import Integer, Literal, Map, Real, Vector

def one_hot_encoding(
    training_points: NDArray,
    test_points: NDArray,
    categorical_indexes: Vector[Integer] = [],
    categories_list: Vector[Vector[Literal]] = []
) -> Tuple[Tuple[NDArray, NDArray], Integer]:
    '''
    Encode categorical features as a one-hot numeric array
    :param training_points: Points in the training set
    :param test_points: Points in the test set
    :param categorical_indexes: Indexes of the categorical features
    :param categories_list: Holds the categories expected in the every categorical feature
    :return: Training and test points with encoded categorical features, and the encoding size
    '''
    if len(categorical_indexes) == 0:
        return (training_points, test_points), 0

    encoding_size = 0

    binary_categorical_features = []
    for feature_index, categories in zip(categorical_indexes, categories_list):
        if len(categories) <= 2:
            binary_categorical_features.append(feature_index)
            encoding_size += 1

    binary_encoder = ColumnTransformer(
        [('', OrdinalEncoder(), binary_categorical_features)]
    )

    one_hot_categories = []
    one_hot_categorical_indexes = []
    for feature_index, categories in zip(categorical_indexes, categories_list):
        if len(categories) > 2:
            one_hot_categories.append(categories)
            one_hot_categorical_indexes.append(feature_index)
            encoding_size += len(categories)

    one_hot_encoder = ColumnTransformer(
        [('', OneHotEncoder(categories=one_hot_categories, sparse=False), one_hot_categorical_indexes)]
    )

    training_points = concatenate((
            binary_encoder.fit_transform(training_points),
            one_hot_encoder.fit_transform(training_points),
            delete(training_points, categorical_indexes, axis=1)
        ), axis=1)
    test_points = concatenate((
            binary_encoder.transform(test_points),
            one_hot_encoder.transform(test_points),
            delete(test_points, categorical_indexes, axis=1),
        ), axis=1)

    return (training_points, test_points), encoding_size

def set_scaler_with_range(
    scaler: MinMaxScaler,
    points: NDArray,
    feature_range: Map[Literal, Tuple[Real, Real]] = {}
) -> None:
    '''
    Fit a scaler to a specific range
    :param scaler: Scaler to be fitted to a specific range
    :param points: Target points
    :param feature_range: Minimum and maximum value of the numerical features of the points
    :return: Training and test points with scaled features
    '''
    if 'all' in feature_range:
        scaler.fit(array([
            [feature_range['all'][0] for _ in range(points.shape[1])],
            [feature_range['all'][1] for _ in range(points.shape[1])]
        ]))
    else:
        scaler.fit(points)
        scaler.fit(array([
            [feature_range[i][0] if i in feature_range else scaler.data_min_[i] for i in range(points.shape[1])],
            [feature_range[i][1] if i in feature_range else scaler.data_max_[i] for i in range(points.shape[1])]
        ]))

def scale_features(
    training_points: NDArray,
    test_points: NDArray,
    feature_range: Map[Literal, Tuple[Real, Real]] = {},
    categorical_indexes: Vector[Integer] = [],
    categories_list: Vector[Vector[Literal]] = []
) -> Tuple[Tuple[NDArray, NDArray], Integer]:
    '''
    Scale the features of training and test points to the [0,1] range
    :param training_points: Points in the training set
    :param test_points: Points in the test set
    :param feature_range: Minimum and maximum value of the numerical features of the points (if available)
    :param categorical_features: What are the categorical features (indexes)
    :param categories_list: Holds the categories expected in the every categorical feature
    :return: Training and test points with scaled features
    '''
    (training_points, test_points), encoded_with_one_hot = one_hot_encoding(
        training_points,
        test_points,
        categorical_indexes,
        categories_list
    )

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    if len(feature_range) == 0:
        if encoded_with_one_hot == 0:
            training_points = scaler.fit_transform(training_points)
            test_points = scaler.transform(test_points)

        else:
            categorical_training_features, numerical_training_features = hsplit(
                training_points, [encoded_with_one_hot]
            )
            training_points = concatenate((
                categorical_training_features, scaler.fit_transform(numerical_training_features)
            ), axis=1)

            categorical_test_features, numerical_test_features = hsplit(
                test_points, [encoded_with_one_hot]
            )
            test_points = concatenate((
                categorical_test_features, scaler.transform(numerical_test_features)
            ), axis=1)
        
    else:
        if encoded_with_one_hot == 0:
            set_scaler_with_range(scaler, training_points, feature_range)
            training_points = scaler.transform(training_points)
            test_points = scaler.transform(test_points)

        else:
            categorical_training_features, numerical_training_features = hsplit(
                training_points, [encoded_with_one_hot]
            )
            set_scaler_with_range(scaler, numerical_training_features, feature_range)
            training_points = concatenate((
                categorical_training_features, scaler.transform(numerical_training_features)
            ), axis=1)

            categorical_test_features, numerical_test_features = hsplit(
                test_points, [encoded_with_one_hot]
            )
            test_points = concatenate((
                categorical_test_features, scaler.transform(numerical_test_features)
            ), axis=1)

    return training_points, test_points