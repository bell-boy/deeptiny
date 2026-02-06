cmake_minimum_required(VERSION 3.23)

add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/doctest)

file(GLOB TEST_UTIL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/test_utils/*.cc"
)

add_library(test_utils STATIC
  ${TEST_UTIL_SOURCES}
)


target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/test_utils)
target_include_directories(test_utils PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_utils PUBLIC deeptiny doctest)
target_compile_features(test_utils PUBLIC cxx_std_23)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cc"
)

foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE) # e.g. test_add

  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE deeptiny doctest test_utils)
  target_compile_features(${test_name} PRIVATE cxx_std_23)
  target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
