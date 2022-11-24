# -*- coding: utf-8 -*-
# =============================================================================
# File: min_heap.py
# Updated: 05/11/2022
# =============================================================================
'''Define a min heap structure'''
# =============================================================================
# Dependencies:
#   ../base.py
# =============================================================================

from collections import deque
from heapq import heapify, heappop
from typing import Any

from robustness import Integer, Vector

class MinHeap:
    '''Represent a min heap structure'''
    
    def __init__(self,
        elements: Vector[Any]
    ) -> None:
        '''
        Let the class initialize the object's attributes
        :param elements: Elements to put in the min heap
        '''
        self.__n_smallest = deque([]) 
        self.__heap = elements
        
        heapify(self.__heap)

    def get_size(self) -> Integer:
        '''
        Return the size of the min heap
        :return: Number of elements in the min heap
        '''            
        return len(self.__n_smallest) + len(self.__heap)

    def get_items_ordered_so_far(self) -> Vector[Any]:
        '''
        Return the items ordered so far
        :return: Items ordered so far in a vector
        '''      
        return self.__n_smallest

    def get_nth_smallest(self,
        n: Integer
    ) -> Any:
        '''
        Return the n-th smallest element in the min heap
        :param n: n-th desired smallest element
        :return: n-th smallest element in the min heap
        '''            
        while n > len(self.__n_smallest):
            self.__n_smallest.append(heappop(self.__heap))
        return self.__n_smallest[n - 1]

    def pop(self) -> Any:
        '''
        Extract the smallest element in the min heap
        :return: The smallest element in the min heap
        '''            
        return self.__n_smallest.popleft() if len(self.__n_smallest) > 0 else heappop(self.__heap)