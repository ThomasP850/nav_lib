## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup, find_packages
from catkin_pkg.python_setup import generate_distutils_setup

# Include nav_lib and all subpackages (nav_lib.planner, nav_lib.motion, nav_lib.pointcloud)
setup_args = generate_distutils_setup(
    packages=find_packages('lib'),
    package_dir={'': 'lib'},
    install_requires=['open3d'],
)

setup(**setup_args)

