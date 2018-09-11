"""
    Runs all tests in this directory

    Note:
        Only test file with names: test*.py are executed

    Version:
        2018-09-11 DWW
"""

import os
from unittest import TestLoader, TextTestRunner

start = os.getcwd()
texts = TestLoader().discover(start)
TextTestRunner().run(texts)
