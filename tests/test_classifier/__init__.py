import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(current_dir, '..')
project_dir = os.path.join(test_dir, '..', '..')

sys.path.insert(0, project_dir)
