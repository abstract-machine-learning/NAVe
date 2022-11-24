# -*- coding: utf-8 -*-
# =============================================================================
# File: setup.py
# Updated: 05/11/2022
# =============================================================================
'''Describe all of the metadata about the project'''
# =============================================================================
# Dependencies: -
# =============================================================================

from setuptools import setup

with open('requirements.txt') as file:
    requirements = file.read().splitlines()

setup(
    name='kNAVe',
    version='1.0',
    install_requires=requirements,
    py_modules=[]
)