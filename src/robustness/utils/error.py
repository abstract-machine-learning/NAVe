# -*- coding: utf-8 -*-
# =============================================================================
# File: error.py
# Updated: 05/11/2022
# =============================================================================
'''Handle errors'''
# =============================================================================
# Dependencies:
#   ../base.py
# =============================================================================

from sys import exit
from robustness import String

class Error:
    '''Show error messages'''

    def __init__(self,
        message: String
    ) -> None:
        '''
        Let the class initialize the object's attributes
        ''' 
        print('\n ERROR: ' + message + '\n')
        exit()