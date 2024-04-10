cmake_minimum_required(VERSION 3.20)

project(alaya VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    fmt
    URL https://github.com/fmtlib/fmt/releases/download/10.2.1/fmt-10.2.1.zip
    DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(fmt)

add_compile_options(-march=native)

add_compile_options(-lsqlite3)

find_package(OpenMP REQUIRED)

add_subdirectory(thirdparty)

add_library(alaya
  STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/kmeans.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/random_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/TableSchema.cpp
)
target_link_libraries(alaya
  PUBLIC
  OpenMP::OpenMP_CXX
  fmt::fmt
  faiss
)
target_include_directories(alaya
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(MKL)
if(MKL_FOUND)
  message(STATUS "MKL found")
  add_compile_definitions(USE_MKL)
  target_link_libraries(alaya PUBLIC MKL::MKL)
else()
  find_package(BLAS)
  add_compile_definitions(USE_BLAS)
  target_link_libraries(alaya PUBLIC ${BLAS_LIBRARIES})
endif()

# add_subdirectory(tests)

# add_executable(test_RQ test_RQ.cpp)
# target_link_libraries(test_RQ alaya)

# add_executable(test_main test_main.cpp)
# target_link_libraries(test_main alaya)

enable_testing()
add_executable(
  distance_test
  tests/distance_test.cpp
)
target_link_libraries(
  distance_test
  alaya
  GTest::gtest_main
)

# add_executable(
#   pq_test
#   tests/pq_test.cpp
# )
# target_link_libraries(
#   pq_test
#   alaya
#   GTest::gtest_main
# )

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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/faiss)

find_package(Boost REQUIRED thread)

add_executable(
  ivf_test
  tests/ivf_test.cpp
)
target_link_libraries(
  ivf_test
  alaya
  GTest::gtest_main
)

add_executable(
  hello_test
  tests/hello_test.cpp
)
target_link_libraries(
  bucket_test
  alaya
  Boost::thread
  pthread
  OpenMP::OpenMP_CXX
  GTest::gtest_main
)


# complete deletion

include(GoogleTest)
gtest_discover_tests(distance_test)
<<<<<<< HEAD
# gtest_discover_tests(pq_test)
gtest_discover_tests(hello_test)

gtest_discover_tests(bucket_test)


=======
gtest_discover_tests(pq_test)
gtest_discover_tests(ivf_test)
gtest_discover_tests(hello_test)
>>>>>>> master
