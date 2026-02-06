cmake_minimum_required(VERSION 3.23)

project(transfomer_demo LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
option(TRANSFOMER_DEMO_BUILD_TESTS "Build transfomer demo tests"
       ${PROJECT_IS_TOP_LEVEL})

set(DEEPTINY_GIT_REPOSITORY "https://github.com/bell-boy/deeptiny.git" CACHE STRING
    "Git repository for Deep Tiny")
set(DEEPTINY_GIT_TAG "154acc95dc69a20646377d3e7059b6f46f8b9a18" CACHE STRING
    "Pinned Deep Tiny commit")

# Keep the demo isolated from Deep Tiny's internal test suite.
set(BUILD_TESTING OFF CACHE BOOL "Disable Deep Tiny tests for demo builds" FORCE)
set(DEEPTINY_BUILD_TESTS OFF CACHE BOOL "Disable Deep Tiny tests for demo builds" FORCE)
set(DEEPTINY_ENABLE_WERROR OFF CACHE BOOL "Disable Werror for embedded demo builds" FORCE)

FetchContent_Declare(
  deeptiny
  GIT_REPOSITORY "${DEEPTINY_GIT_REPOSITORY}"
  GIT_TAG "${DEEPTINY_GIT_TAG}"
  GIT_SHALLOW FALSE
)

FetchContent_MakeAvailable(deeptiny)

add_library(transfomer_demo_modules STATIC
  src/modules/embedding.cc
)
target_compile_features(transfomer_demo_modules PUBLIC cxx_std_23)
target_include_directories(transfomer_demo_modules PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(transfomer_demo_modules PUBLIC deeptiny)

add_executable(transfomer_demo src/main.cc)

target_compile_features(transfomer_demo PRIVATE cxx_std_23)
target_link_libraries(transfomer_demo PRIVATE transfomer_demo_modules deeptiny)

if (TRANSFOMER_DEMO_BUILD_TESTS)
  enable_testing()
  add_executable(transfomer_demo_embedding_test tests/embedding_test.cc)
  target_compile_features(transfomer_demo_embedding_test PRIVATE cxx_std_23)
  target_link_libraries(transfomer_demo_embedding_test
    PRIVATE transfomer_demo_modules deeptiny
  )
  target_include_directories(transfomer_demo_embedding_test PRIVATE
    ${deeptiny_SOURCE_DIR}/src
  )
  add_test(
    NAME transfomer_demo_embedding_test
    COMMAND transfomer_demo_embedding_test
  )
endif()
