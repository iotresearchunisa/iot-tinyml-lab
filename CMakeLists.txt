cmake_minimum_required(VERSION 3.10)
project(AifesIoTLab C)

set(CMAKE_C_STANDARD 99)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(AIFES_PATH "${PROJECT_ROOT}/AIfES_for_Arduino-main/src")

# Collect all AIfES source files
file(GLOB_RECURSE AIFES_SRC "${AIFES_PATH}/*.c")

# Include directories
include_directories(
        ${AIFES_PATH}
        ${PROJECT_ROOT}
)

add_library(AIfES STATIC ${AIFES_SRC})

add_executable(AifesIoTLab main.c)
target_link_libraries(AifesIoTLab PRIVATE AIfES m)