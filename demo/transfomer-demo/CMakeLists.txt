cmake_minimum_required(VERSION 3.23)

project(transfomer_demo LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

set(DEEPTINY_GIT_REPOSITORY "https://github.com/bell-boy/deeptiny.git" CACHE STRING
    "Git repository for Deep Tiny")
set(DEEPTINY_GIT_TAG "04d92a7abd1a4932e642ae5ff2a3cfd9f4fa0abd" CACHE STRING
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

add_executable(transfomer_demo src/main.cc)

target_compile_features(transfomer_demo PRIVATE cxx_std_23)
target_link_libraries(transfomer_demo PRIVATE deeptiny)
