cmake_minimum_required(VERSION 3.20)

project(alaya VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(alaya
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/kmeans.cpp
)

target_include_directories(alaya
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include_directories(src)