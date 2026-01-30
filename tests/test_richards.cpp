#include <catch2/catch_test_macros.hpp>
#include "dgw/dgw.hpp"

using namespace dgw;

TEST_CASE("Richards3D solver creation", "[richards]") {
    auto physics = create_physics(GoverningEquation::Richards3D);
    REQUIRE(physics != nullptr);
    REQUIRE(physics->type() == GoverningEquation::Richards3D);
}
