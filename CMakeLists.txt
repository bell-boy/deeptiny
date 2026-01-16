cmake_minimum_required(VERSION 3.23)

project(deeptiny)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


file(GLOB DEEPTINY_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

add_library(deeptiny STATIC
  ${DEEPTINY_SOURCES}
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/storage.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/kernels.cc"
)

target_compile_features(deeptiny PUBLIC cxx_std_23)

target_include_directories(deeptiny
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(OpenBLAS REQUIRED)

if (TARGET OpenBLAS::OpenBLAS)
  target_link_libraries(deeptiny
    PUBLIC OpenBLAS::OpenBLAS
  )
else()
  if (NOT OpenBLAS_INCLUDE_DIRS OR NOT OpenBLAS_LIBRARIES)
    message(FATAL_ERROR "OpenBLAS found but does not define OpenBLAS::OpenBLAS or required variables.")
  endif()
  target_include_directories(deeptiny
    PUBLIC ${OpenBLAS_INCLUDE_DIRS}
  )
  target_link_libraries(deeptiny
    PUBLIC ${OpenBLAS_LIBRARIES}
  )
endif()

if (MSVC)
  target_compile_options(deeptiny PRIVATE /W4)
else()
  target_compile_options(deeptiny PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
