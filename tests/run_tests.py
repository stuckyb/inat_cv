#!/usr/bin/python3

# Copyright (C) 2021 Brian J. Stucky
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# A simple test runner that provides a minimal interface for running tests,
# including automatic test discovery, that produces verbose output by default
# and is called the same way in Python 2.7 and Python 3.
#
# What this script does (without additional arguments) is the same as calling
#
# $ python3 -m unittest -v
#
# for Python 3 and
#
# $ python -m unittest -v discover
#
# for Python 2.7.
#
# You can also test specific test modules using this script, e.g.
#
# $ python3 run_tests.py TestModule
#


import unittest
import sys
import os.path


# Make sure we can find the source modules.
source_dir = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../'
    )
)
sys.path.append(source_dir)

if sys.version_info[0] == 3:
    # For Python 3.
    unittest.main(module=None, argv=(sys.argv + ['-v']))
else:
    # For Python 2.7.

    # See if any modules or test cases were explicitly referenced.
    testcnt = 0
    for argval in sys.argv[1:]:
        if not(argval.startswith('-')):
            testcnt += 1

    if testcnt > 0:
        newargv = sys.argv + ['-v']
    else:
        newargv = sys.argv + ['discover', '-v']

    unittest.main(module=None, argv=newargv)

