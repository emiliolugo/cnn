cmake_minimum_required(VERSION 3.16)

project(cnn_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cnn_app
    src/main.cpp
)

target_include_directories(cnn_app PRIVATE include)
