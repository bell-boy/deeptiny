cmake_minimum_required(VERSION 3.23)

project(transfomer_demo LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
include(CTest)
option(TRANSFOMER_DEMO_BUILD_TESTS "Build transfomer-demo tests" ON)
option(TRANSFOMER_DEMO_ENABLE_TOKENIZERS_CPP
       "Build transfomer-demo with tokenizers-cpp submodule" ON)

set(DEEPTINY_GIT_REPOSITORY "https://github.com/bell-boy/deeptiny.git" CACHE STRING
    "Git repository for Deep Tiny")
set(DEEPTINY_GIT_TAG "7e5284e8e2103f029ce791dbe08b6c4847a568ac" CACHE STRING
    "Pinned Deep Tiny commit")
set(TRANSFOMER_DEMO_TOKENIZERS_CPP_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp" CACHE PATH
    "Path to tokenizers-cpp submodule source")

# Keep the demo isolated from Deep Tiny's internal test suite.
set(_demo_build_testing "${TRANSFOMER_DEMO_BUILD_TESTS}")
set(BUILD_TESTING OFF CACHE BOOL
    "Disable Deep Tiny tests while configuring dependency" FORCE)
set(DEEPTINY_BUILD_TESTS OFF CACHE BOOL "Disable Deep Tiny tests for demo builds" FORCE)
set(DEEPTINY_ENABLE_WERROR OFF CACHE BOOL "Disable Werror for embedded demo builds" FORCE)

FetchContent_Declare(
  deeptiny
  GIT_REPOSITORY "${DEEPTINY_GIT_REPOSITORY}"
  GIT_TAG "${DEEPTINY_GIT_TAG}"
  GIT_SHALLOW FALSE
)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY "https://github.com/nlohmann/json.git"
  GIT_TAG "v3.11.3"
  GIT_SHALLOW TRUE
)

if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_DEEPTINY)
  get_filename_component(_local_deeptiny_root "${CMAKE_CURRENT_SOURCE_DIR}/../.."
                         ABSOLUTE)
  if(EXISTS "${_local_deeptiny_root}/include/deeptiny/functional.h"
     AND EXISTS "${_local_deeptiny_root}/src")
    set(FETCHCONTENT_SOURCE_DIR_DEEPTINY "${_local_deeptiny_root}" CACHE PATH
        "Local Deep Tiny source override for demo development")
  endif()
endif()

FetchContent_MakeAvailable(deeptiny nlohmann_json)
set(BUILD_TESTING "${_demo_build_testing}" CACHE BOOL
    "Build transfomer-demo tests" FORCE)

if(TRANSFOMER_DEMO_ENABLE_TOKENIZERS_CPP)
  if(NOT EXISTS "${TRANSFOMER_DEMO_TOKENIZERS_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "tokenizers-cpp submodule not found at ${TRANSFOMER_DEMO_TOKENIZERS_CPP_DIR}. "
      "Run `git submodule update --init --recursive` or disable "
      "TRANSFOMER_DEMO_ENABLE_TOKENIZERS_CPP.")
  endif()

  if(NOT EXISTS "${TRANSFOMER_DEMO_TOKENIZERS_CPP_DIR}/msgpack/CMakeLists.txt"
     OR NOT EXISTS "${TRANSFOMER_DEMO_TOKENIZERS_CPP_DIR}/sentencepiece/CMakeLists.txt")
    message(FATAL_ERROR
      "tokenizers-cpp nested submodules are missing. "
      "Run `git submodule update --init --recursive`.")
  endif()

  find_program(TRANSFOMER_DEMO_CARGO_EXECUTABLE cargo)
  if(NOT TRANSFOMER_DEMO_CARGO_EXECUTABLE)
    message(FATAL_ERROR
      "tokenizers-cpp requires cargo (Rust). Install Rust or disable "
      "TRANSFOMER_DEMO_ENABLE_TOKENIZERS_CPP.")
  endif()

  add_subdirectory("${TRANSFOMER_DEMO_TOKENIZERS_CPP_DIR}"
                   "${CMAKE_CURRENT_BINARY_DIR}/tokenizers-cpp"
                   EXCLUDE_FROM_ALL)
endif()

add_executable(
  transfomer_demo
  src/main.cc
  src/transformer.cc
  src/smollm2_135m_instruct_loader.cc
)

add_executable(
  transfomer_demo_benchmark
  src/benchmark.cc
  src/transformer.cc
  src/smollm2_135m_instruct_loader.cc
)

function(configure_transfomer_demo_target target_name)
  target_compile_features(${target_name} PRIVATE cxx_std_23)
  target_link_libraries(${target_name} PRIVATE deeptiny nlohmann_json::nlohmann_json)
  if(TRANSFOMER_DEMO_ENABLE_TOKENIZERS_CPP)
    target_compile_definitions(${target_name}
                               PRIVATE TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP=1)
    target_link_libraries(${target_name} PRIVATE tokenizers_cpp)
  endif()
endfunction()

configure_transfomer_demo_target(transfomer_demo)
configure_transfomer_demo_target(transfomer_demo_benchmark)

target_compile_options(transfomer_demo_benchmark PRIVATE -pg)
target_link_options(transfomer_demo_benchmark PRIVATE -pg)
