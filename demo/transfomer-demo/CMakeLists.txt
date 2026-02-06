cmake_minimum_required(VERSION 3.23)

project(transfomer_demo LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
include(CTest)
option(TRANSFOMER_DEMO_BUILD_TESTS "Build transfomer-demo tests" ON)

set(DEEPTINY_GIT_REPOSITORY "https://github.com/bell-boy/deeptiny.git" CACHE STRING
    "Git repository for Deep Tiny")
set(DEEPTINY_GIT_TAG "154acc95dc69a20646377d3e7059b6f46f8b9a18" CACHE STRING
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

add_library(transfomer_modules STATIC
  src/modules/linear.cc
  src/modules/llama_mlp.cc
  src/modules/embedding.cc
)
target_compile_features(transfomer_modules PUBLIC cxx_std_23)
target_include_directories(transfomer_modules PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(transfomer_modules PUBLIC deeptiny)

add_executable(transfomer_demo src/main.cc)

target_compile_features(transfomer_demo PRIVATE cxx_std_23)
target_link_libraries(transfomer_demo PRIVATE transfomer_modules)

if(TRANSFOMER_DEMO_BUILD_TESTS)
  enable_testing()

  add_executable(modules_smoke tests/modules_smoke.cc)
  target_compile_features(modules_smoke PRIVATE cxx_std_23)
  target_link_libraries(modules_smoke PRIVATE transfomer_modules)
  target_include_directories(modules_smoke PRIVATE ${deeptiny_SOURCE_DIR}/src)
  add_test(NAME modules_smoke COMMAND modules_smoke)

  add_executable(transfomer_demo_embedding_test tests/embedding_test.cc)
  target_compile_features(transfomer_demo_embedding_test PRIVATE cxx_std_23)
  target_link_libraries(transfomer_demo_embedding_test PRIVATE transfomer_modules)
  target_include_directories(transfomer_demo_embedding_test PRIVATE
    ${deeptiny_SOURCE_DIR}/src
  )
  add_test(
    NAME transfomer_demo_embedding_test
    COMMAND transfomer_demo_embedding_test
  )
endif()
