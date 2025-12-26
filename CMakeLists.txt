cmake_minimum_required(VERSION 3.23)

project(deeptiny)

file(GLOB DEEPTINY_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

add_library(deeptiny STATIC
  ${DEEPTINY_SOURCES}
)

target_compile_features(deeptiny PUBLIC cxx_std_20)

target_include_directories(deeptiny
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (MSVC)
  target_compile_options(deeptiny PRIVATE /W4)
else()
  target_compile_options(deeptiny PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()