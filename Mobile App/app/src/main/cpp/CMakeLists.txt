cmake_minimum_required(VERSION 3.22.1)

project("llama-android")

# Options
option(GGML_USE_CPU "Use CPU backend" ON)
option(LLAMA_MMAP "Enable mmap" ON)

add_compile_definitions(GGML_USE_CPU=1 _POSIX_MAPPED_FILES)
add_compile_definitions(GGML_VERSION="0.1.0")
add_compile_definitions(GGML_COMMIT="unknown")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mtmd)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor)

# Find sources
file(GLOB GGML_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/*.cpp")
file(GLOB GGML_CPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu/*.cpp")
file(GLOB GGML_ARM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu/arch/arm/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu/arch/arm/*.cpp")
list(APPEND GGML_SOURCES ${GGML_CPU_SOURCES} ${GGML_ARM_SOURCES})

file(GLOB_RECURSE LLAMA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp")
list(APPEND COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-httplib/httplib.cpp")

file(GLOB MTMD_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/mtmd/*.cpp")
list(FILTER MTMD_SOURCES EXCLUDE REGEX "mtmd-cli.cpp")
file(GLOB MTMD_MODELS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/mtmd/models/*.cpp")
list(APPEND MTMD_SOURCES ${MTMD_MODELS_SOURCES})

add_library(
        llama-android
        SHARED
        llama-android.cpp
        ${GGML_SOURCES}
        ${LLAMA_SOURCES}
        ${COMMON_SOURCES}
        ${MTMD_SOURCES}
)

find_library(
        log-lib
        log
)

target_link_libraries(
        llama-android
        ${log-lib}
)
