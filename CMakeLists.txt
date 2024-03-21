cmake_minimum_required(VERSION 3.20)

project(alaya VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_program(CLANG_EXECUTABLE clang-17)  
# if(NOT CLANG_EXECUTABLE)  
#     message(FATAL_ERROR "Clang 17 not found")  
# endif()

# set(CMAKE_CXX_COMPILER ${CLANG_EXECUTABLE})

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-stdlib=libc++) 
  add_link_options(-stdlib=libc++)
  link_libraries(c++)
endif()

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    fmt
    URL https://github.com/fmtlib/fmt/releases/download/10.2.1/fmt-10.2.1.zip
)
FetchContent_MakeAvailable(fmt)

add_compile_options(-march=native)

find_package(OpenMP REQUIRED)

add_library(alaya
  STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/kmeans.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/random_utils.cpp
)
target_link_libraries(alaya
  PUBLIC
  OpenMP::OpenMP_CXX
  fmt
)

target_include_directories(alaya
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


enable_testing()
add_executable(distance_test tests/distance_test.cpp)
target_link_libraries(distance_test alaya GTest::gtest_main)

add_executable(
  hello_test
  tests/hello_test.cpp
)
target_link_libraries(
  hello_test
  GTest::gtest_main
)
#TODO delete this
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(
  pool_test
  tests/pool_test.cpp
)
target_link_libraries(
  pool_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(distance_test)
gtest_discover_tests(hello_test)


# add_subdirectory(tests)
