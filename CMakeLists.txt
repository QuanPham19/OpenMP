cmake_minimum_required(VERSION 3.16)
project(omp_practice LANGUAGES CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimized build so timings mean something
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

# add_executable(test src/test.cpp)
# target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX)
# target_compile_options(test PRIVATE -O3)

add_executable(vector_add 
  # src/vector_add.cpp
  src/matrix_multiply.cpp)
target_link_libraries(vector_add PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(vector_add PRIVATE -O3 -march=native -fopenmp)