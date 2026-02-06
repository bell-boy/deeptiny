cmake_minimum_required(VERSION 3.23)

project(deeptiny)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(DEEPTINY_PACKAGE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/deeptiny")
set(DEEPTINY_PACKAGE_VERSION "0.0.0")

file(GLOB DEEPTINY_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
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
    PRIVATE ${OpenBLAS_INCLUDE_DIRS}
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

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/deeptinyConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/deeptinyConfig.cmake"
  INSTALL_DESTINATION "${DEEPTINY_PACKAGE_INSTALL_DIR}"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/deeptinyConfigVersion.cmake"
  VERSION "${DEEPTINY_PACKAGE_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(TARGETS deeptiny
  EXPORT deeptinyTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(EXPORT deeptinyTargets
  FILE deeptinyTargets.cmake
  NAMESPACE deeptiny::
  DESTINATION "${DEEPTINY_PACKAGE_INSTALL_DIR}"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/deeptinyConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/deeptinyConfigVersion.cmake"
  DESTINATION "${DEEPTINY_PACKAGE_INSTALL_DIR}"
)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
