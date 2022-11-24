# -*- coding: utf-8 -*-
# =============================================================================
# File: distances.py
# Updated: 05/11/2022
# =============================================================================
'''Contain the most used distance functions'''
# =============================================================================
# Dependencies:
#   ../base.py
# =============================================================================

from typing import Any

from robustness import String, Vector

def compute_distance(
    point1:Vector[Any],
    point2: Vector[Any],
    distance_metric: String,
    init: Any = 0.0
) -> Any: 
    '''
    Compute the distance between two points
    :param point1: First point involved in the distance computation
    :param point2: Second point involved in the distance computation
    :param distance_metric: Desired distance metric
    :param init: Initial value. The result is init + dist(poin1, point2)
    :return: Result of the distance computation
    '''
    if distance_metric == 'manhattan':
        return manhattan_distance(point1, point2, init)
    elif distance_metric == 'euclidean':
        return squared_euclidean_distance(point1, point2, init)
    else:
        raise Exception('\nUnsupported distance metric')

def manhattan_distance(
    point1: Vector[Any],
    point2: Vector[Any],
    init: Any = 0.0
) -> Any: 
    '''
    Compute the Manhattan distance between two points
    :param point1: First point involved in the distance computation
    :param point2: Second point involved in the distance computation
    :param init: Initial value. The result is init + dist(poin1, point2)
    :return: Result of the Manhattan distance computation
    '''
    distance = init
    for i in range(len(point1)):
        distance = distance + abs(point1[i] - point2[i])

    return distance

def squared_euclidean_distance(
    point1: Vector[Any],
    point2: Vector[Any],
    init: Any = 0.0
) -> Any: 
    '''
    Compute the Euclidean distance between two points, without the square root at the end
    :param point1: First point involved in the distance computation
    :param point2: Second point involved in the distance computation
    :param init: Initial value. The result is init + dist(poin1, point2)
    :return: Result of the squared Euclidean distance computation
    '''
    distance = init
    for i in range(len(point1)):
        distance = distance + (point1[i] - point2[i]) ** 2
        
    return distance