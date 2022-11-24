# -*- coding: utf-8 -*-
# =============================================================================
# File: __init__.py
# Updated: 05/11/2022
# =============================================================================
'''Make Python treat directories containing the file as packages'''
# =============================================================================
# Dependencies:
#   ./abstract_classifier.py
#   ./interval_classifier.py
# =============================================================================

from .abstract_classifier import AbstractClassifier
from .interval_classifier import IntervalClassifier

__all__ = [
    'AbstractClassifier',
    'IntervalClassifier',
]