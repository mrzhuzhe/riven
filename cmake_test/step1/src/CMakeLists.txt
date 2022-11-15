cmake_minimum_required(VERSION 3.21.2)
project(Tutorial)

set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

# configure a header file to pass some of the cmake setting to source code
configure_file(
    "${PROJECT_SOURCE_DIR}/TutorialConfig.h.in"
    "${PROJECT_SOURCE_DIR}/TutorialConfig.h"
)

# add the binary tree to the search path for include files so that we will find TutorialConfig.h
#include_directories("$PROJECT_BINARY_DIR")

add_executable(Tutorial tutorial.cpp)