"""
    Runs all tests in this directory

    Note:
        Only test file with names: test*.py are executed

    Version:
        2018-09-12 DWW
"""

import os
from unittest import TestLoader, TextTestRunner

start = os.getcwd()
suite = TestLoader().discover(start)
TextTestRunner().run(suite)
