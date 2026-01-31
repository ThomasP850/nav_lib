## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Include all subpackages (nav_lib, nav_lib.motion, nav_lib.planner, ...)
setup_args = generate_distutils_setup(
    packages=['nav_lib'],
    package_dir={'': 'lib'},
    install_requires=['open3d'],
)

setup(**setup_args)

