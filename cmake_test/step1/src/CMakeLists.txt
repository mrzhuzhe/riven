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
include_directories("$PROJECT_BINARY_DIR")

option (USE_MYMATH  "Use tutorial provided math implementation" ON)


add_library(MathFunctions mysqrt.cpp)
#add_library(MathFunctions SHARED mysqrt.cpp)

if (USE_MYMATH)
    include_directories ("${PROJECT_SOURCE_DIR}/MathFunctions")
    add_subdirectory(MathFunctions)
    set (EXTRA_LIBS ${EXTRA_LIBS} MathFunctions)
endif (USE_MYMATH)

add_executable(Tutorial tutorial.cpp)
target_link_libraries(Tutorial ${EXTRA_LIBS})


#install (TARGETS Tutorial DESTINATION bin)

#install (FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h" DESTINATION include)
