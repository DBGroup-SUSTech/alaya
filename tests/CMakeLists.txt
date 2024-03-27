enable_testing()
add_executable(
  distance_test
  distance_test.cpp
)
target_link_libraries(
  distance_test
  alaya
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(distance_test)

# add_executable(
#   hello_test
#   hello_test.cc
# )
# target_link_libraries(
#   hello_test
#   GTest::gtest_main
# )

# include(GoogleTest)
# gtest_discover_tests(hello_test)