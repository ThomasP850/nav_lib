cmake_minimum_required(VERSION 3.0.2)
project(nav_lib)


catkin_python_setup()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package()