cmake_minimum_required(VERSION 3.23)

project(transfomer_demo LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
include(CTest)
option(TRANSFOMER_DEMO_BUILD_TESTS "Build transfomer-demo tests" ON)

set(DEEPTINY_GIT_REPOSITORY "https://github.com/bell-boy/deeptiny.git" CACHE STRING
    "Git repository for Deep Tiny")
set(DEEPTINY_GIT_TAG "7e5284e8e2103f029ce791dbe08b6c4847a568ac" CACHE STRING
    "Pinned Deep Tiny commit")

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

if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_DEEPTINY)
  get_filename_component(_local_deeptiny_root "${CMAKE_CURRENT_SOURCE_DIR}/../.."
                         ABSOLUTE)
  if(EXISTS "${_local_deeptiny_root}/include/deeptiny/functional.h"
     AND EXISTS "${_local_deeptiny_root}/src")
    set(FETCHCONTENT_SOURCE_DIR_DEEPTINY "${_local_deeptiny_root}" CACHE PATH
        "Local Deep Tiny source override for demo development")
  endif()
endif()

FetchContent_MakeAvailable(deeptiny)
set(BUILD_TESTING "${_demo_build_testing}" CACHE BOOL
    "Build transfomer-demo tests" FORCE)

add_executable(
  transfomer_demo
  src/main.cc
  src/smollm2_135m_instruct_loader.cc
)

target_compile_features(transfomer_demo PRIVATE cxx_std_23)
target_link_libraries(transfomer_demo PRIVATE deeptiny)
