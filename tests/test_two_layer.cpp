#include <catch2/catch_test_macros.hpp>
#include "dgw/dgw.hpp"

using namespace dgw;

TEST_CASE("TwoLayer solver creation", "[two_layer]") {
    auto physics = create_physics(GoverningEquation::TwoLayer);
    REQUIRE(physics != nullptr);
    REQUIRE(physics->type() == GoverningEquation::TwoLayer);
}
