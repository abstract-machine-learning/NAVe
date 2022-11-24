# -*- coding: utf-8 -*-
# =============================================================================
# File: dataset.py
# Updated: 05/11/2022
# =============================================================================
'''Defines the structure of a dataset'''
# =============================================================================
# Dependencies:
#   ./base.py
# =============================================================================

from robustness import Integer, Literal, Real, Set, Vector

class Dataset:
    '''Represents a dataset'''
    
    def __init__(self,
        points: Vector[Vector[Real]],
        labels: Vector[Literal]
    ) -> None:
        '''
        Lets the class initialize the object's attributes
        :param points: Data points
        :param labels: Labels associated with each point
        '''
        self.__points = points
        self.__labels = labels
        self.__classes = sorted([*Set(labels)])
    
    def get_points(self) -> Vector[Vector[Real]]:
        '''
        Returns the points in the dataset
        :return: All points in the dataset
        '''
        return self.__points
        
    def get_labels(self) -> Vector[Literal]:
        '''
        Returns the labels in the dataset
        :return: All labels in the dataset
        '''
        return self.__labels

    def get_classes(self) -> Vector[Literal]:
        '''
        Returns the classess in the dataset
        :return: All classess in the dataset
        '''
        return self.__classes

    def num_points(self) -> Integer:
        '''
        Returns the number of points in the dataset
        :return: Number of points in the dataset
        '''
        return len(self.__points)

    def num_features(self) -> Integer:
        '''
        Returns the number of features of a point
        :return: Number of features in the dataset
        '''
        return len(self.__points[0])

    def num_classes(self) -> Integer:
        '''
        Returns the numeber of classes in the dateset
        :return: Numeber of classes in the dataset
        '''
        return len(self.__classes)

    def print_info(self) -> None:
        '''
        Prints the dataset information
        '''
        print('# of points:\t', self.num_points())
        print('# of features:\t', self.num_features()) 
        print('# of classes:\t', self.num_classes())