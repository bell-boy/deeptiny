cmake_minimum_required(VERSION 3.23)

project(deeptiny)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DEEPTINY_BUILD_TESTS "Build deeptiny tests" ${PROJECT_IS_TOP_LEVEL})
option(DEEPTINY_ENABLE_WERROR "Treat warnings as errors for deeptiny"
       ${PROJECT_IS_TOP_LEVEL})

file(GLOB DEEPTINY_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/*.cc"
)

add_library(deeptiny STATIC
  ${DEEPTINY_SOURCES}
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/storage.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/kernels.cc"
)
add_library(deeptiny::deeptiny ALIAS deeptiny)

target_compile_features(deeptiny PUBLIC cxx_std_23)

target_include_directories(deeptiny
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(OpenBLAS REQUIRED CONFIG)

if (TARGET OpenBLAS::OpenBLAS)
  target_link_libraries(deeptiny
    PUBLIC OpenBLAS::OpenBLAS
  )
else()
  if (NOT OpenBLAS_INCLUDE_DIRS OR NOT OpenBLAS_LIBRARIES)
    message(FATAL_ERROR "OpenBLAS found but does not define OpenBLAS::OpenBLAS or required variables.")
  endif()
  target_include_directories(deeptiny
    PRIVATE ${OpenBLAS_INCLUDE_DIRS}
  )
  target_link_libraries(deeptiny
    PUBLIC ${OpenBLAS_LIBRARIES}
  )
endif()

if (MSVC)
  target_compile_options(deeptiny PRIVATE /W4)
  if (DEEPTINY_ENABLE_WERROR)
    target_compile_options(deeptiny PRIVATE /WX)
  endif()
else()
  target_compile_options(deeptiny PRIVATE -Wall -Wextra -Wpedantic)
  if (DEEPTINY_ENABLE_WERROR)
    target_compile_options(deeptiny PRIVATE -Werror)
  endif()
endif()

if(DEEPTINY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
