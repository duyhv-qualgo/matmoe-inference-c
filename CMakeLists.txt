cmake_minimum_required(VERSION 3.20)
project(matmoe_slm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- TFLite ----
# Pull in the in-tree TensorFlow Lite build (it FetchContent's the TF repo).
set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/litert/tflite"
    CACHE PATH "Path to TFLite source directory")

add_subdirectory("${TFLITE_SOURCE_DIR}"
                 "${CMAKE_CURRENT_BINARY_DIR}/tflite_build"
                 EXCLUDE_FROM_ALL)

# ---- Engine library ----
add_library(slm_engine
    src/slm_engine.cpp
)
target_include_directories(slm_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(slm_engine
    PUBLIC tensorflow-lite
)
target_compile_options(slm_engine PRIVATE
    -Wall -Wextra -Wno-unused-parameter -O3
)

# ---- Smoke test binary ----
add_executable(test_slm
    src/test_main.cc
)
target_link_libraries(test_slm
    PRIVATE slm_engine
)
