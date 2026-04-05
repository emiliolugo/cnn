cmake_minimum_required(VERSION 3.16)

project(cnn_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cnn_app
    src/main.cpp
    src/utils/tensor.cpp
)

target_include_directories(cnn_app PRIVATE include)

# Tests
enable_testing()

add_executable(test_tensor
    test/test_tensor.cpp
    src/utils/tensor.cpp
)

target_include_directories(test_tensor PRIVATE include)

add_test(NAME tensor_tests COMMAND test_tensor)

include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(benchmark)

add_executable(bench_tensor
    bench/bench_tensor.cpp
    src/utils/tensor.cpp
)
target_include_directories(bench_tensor PRIVATE include)
target_link_libraries(bench_tensor PRIVATE benchmark::benchmark)