# CMakeLists.txt
# ==============
# Build system for Project 3 — Image Filtering Pipeline.
#
# How to build:
#   mkdir build && cd build
#   cmake ..
#   make -j$(nproc)
#
# How to find your GPU architecture (sm_XX):
#   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
#   e.g.  "8.6"  → sm_86   (RTX 30xx)
#         "7.5"  → sm_75   (RTX 20xx)
#         "8.9"  → sm_89   (RTX 40xx)

cmake_minimum_required(VERSION 3.18)
project(image_filter LANGUAGES CXX CUDA)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Build type ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ---- OpenCV ----
find_package(OpenCV REQUIRED)

# ---- OpenMP ----
find_package(OpenMP REQUIRED)

# ---- CUDA standard ----
set(CMAKE_CUDA_STANDARD 17)

# ---- GPU Architecture ----
# "native" tells nvcc to detect the GPU in the current machine automatically.
# If you are cross-compiling (building on a machine without a GPU), replace
# "native" with the sm value for your target GPU:
#   set(CMAKE_CUDA_ARCHITECTURES 75)   # RTX 20xx
#   set(CMAKE_CUDA_ARCHITECTURES 86)   # RTX 30xx
#   set(CMAKE_CUDA_ARCHITECTURES 89)   # RTX 40xx
set(CMAKE_CUDA_ARCHITECTURES 75)   # RTX 30xx (sm_86)

# ---- Source files ----
set(SOURCES
    src/main.cpp
    src/filters_serial.cpp
    src/filters_omp.cpp
    src/image_io.cpp
    src/filters_cuda.cu
)

# ---- Executable ----
add_executable(image_filter ${SOURCES})

# ---- Link libraries ----
target_link_libraries(image_filter
    PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    cudart
)

# ---- Include paths ----
target_include_directories(image_filter
    PRIVATE
    src/
    ${OpenCV_INCLUDE_DIRS}
)

# ---- CUDA compiler flags ----
target_compile_options(image_filter
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math      # faster math intrinsics (fine for image processing)
        -lineinfo            # keeps line info for profiling with Nsight/nvprof
    >
)
