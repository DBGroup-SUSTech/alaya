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

add_compile_options(-march=native)

find_package(OpenMP REQUIRED)

add_library(alaya
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/kmeans.cpp
)
target_link_libraries(alaya
    PUBLIC
    OpenMP::OpenMP_CXX
)

target_include_directories(alaya
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include_directories(src)

add_executable(test_main test_main.cpp)
target_link_libraries(test_main alaya)