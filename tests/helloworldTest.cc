#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

TEST_CASE("Making sure CI/CD works.") {
  INFO("hello world :)");
  CHECK(10 == 5 * 2);
}
