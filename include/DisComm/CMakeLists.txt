cmake_minimum_required(VERSION 3.0.0)
project(DisComm VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

# ------------------------------------------------------------------------------
# MPI (https://cmake.org/cmake/help/v3.0/module/FindMPI.html)
# ------------------------------------------------------------------------------
# set(MPIEXEC_EXECUTABLE /usr/local/mpich-3.4.3/bin)
# set(MPI_HOME /usr/local/mpich-3.4.3)
# set(MPIEXEC_EXECUTABLE /usr/local/mpich-3.4.3/bin/mpiexec)
# include(/home/dongjiang/jd/rdma-rpc/cmake/FindMPI.cmake)
find_package(MPI REQUIRED)
if(MPI_FOUND)
    message(STATUS "[Find MPI]: YES, and version is ${MPI_VERSION}")
    message(STATUS "\t- The MPI Include - origin: ${MPI_INCLUDE_PATH}")

    message(STATUS "\t- The MPI Include: ${MPI_INCLUDE_DIR}")

    include_directories(${MPI_INCLUDE_PATH})  #如果不添加可能找不到<mpi.h>
    message(STATUS "\t- The MPI Lib - origin: ${MPI_CXX_LIBRARIES}")

    message(STATUS "\t- The MPI Lib: ${MPI_LIB}")
else(MPI_FOUND)
	message(FATAL_ERROR "[Find MPI]: NO")
endif(MPI_FOUND)

add_executable(DisComm main.cpp)
target_link_libraries(DisComm ${MPI_LIBRARIES} alaya)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
