cmake_minimum_required(VERSION 3.23)

project(transfomer_demo LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DEEPTINY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH
    "Path to the Deep Tiny source tree")

if (NOT EXISTS "${DEEPTINY_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
          "DEEPTINY_SOURCE_DIR does not point to a Deep Tiny source tree: ${DEEPTINY_SOURCE_DIR}")
endif()

# Keep the demo isolated from Deep Tiny's internal test suite.
set(BUILD_TESTING OFF CACHE BOOL "Disable Deep Tiny tests for demo builds" FORCE)

add_subdirectory("${DEEPTINY_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/deeptiny")

add_executable(transfomer_demo src/main.cc)

target_compile_features(transfomer_demo PRIVATE cxx_std_23)
target_link_libraries(transfomer_demo PRIVATE deeptiny)
