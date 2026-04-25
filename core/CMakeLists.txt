cmake_minimum_required(VERSION 3.14)
project(CoarseEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories(${NUMPY_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})

add_library(_coarse MODULE coarse_engine.cpp)

find_package(pybind11 CONFIG REQUIRED)

target_link_libraries(_coarse PRIVATE pybind11::module)

if(APPLE)
    target_compile_options(_coarse PRIVATE -O3 -fvisibility=hidden -undefined dynamic_lookup)
endif()

set_target_properties(_coarse PROPERTIES PREFIX "" SUFFIX ".so")
