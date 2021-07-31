cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0054 NEW)
project(HSGSFilter)
set(CMAKE_CXX_STANDARD 17)

# Global Config
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(CUDA REQUIRED 10)
if(CUDA_FOUND)
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
endif()

add_executable(gsfilter gsfilter.cpp)
target_link_libraries(gsfilter)
target_include_directories(gsfilter PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(gsfilter ${CUDA_LIBRARY})
