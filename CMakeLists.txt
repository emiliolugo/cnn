cmake_minimum_required(VERSION 3.16)

project(cnn_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cnn_app
    src/main.cpp
    src/utils/matrix.cpp
)

target_include_directories(cnn_app PRIVATE include)

# Tests
enable_testing()

add_executable(test_matrix
    test/test_matrix.cpp
    src/utils/matrix.cpp
)

target_include_directories(test_matrix PRIVATE include)

add_test(NAME matrix_tests COMMAND test_matrix)