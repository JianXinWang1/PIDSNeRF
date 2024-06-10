cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(demo)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.3/bin/nvcc)
enable_language(CUDA)

set(CMAKE_BUILD_TYPE DEBUG)

set(Torch_DIR /usr/local/libtorch/share/cmake/Torch)
MESSAGE ("${Torch_INCLUDE_DIRS}")
find_package(Torch REQUIRED)


set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(PROJECT_INCLUDE_DIR /home/wjx/ngp_pl_master/models/csrc/include)
include_directories(
    /usr/include/python3.8,
    ${Torch_INCLUDE_DIRS}, 
    ${CUDA_INCLUDE_DIRS},
    ${PROJECT_INCLUDE_DIR}
    
)

set(SRC_LIST ./test.cu)
add_executable(demo ${SRC_LIST})

target_link_libraries(demo ${TORCH_LIBRARIES})
target_link_libraries(demo  /usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so)

