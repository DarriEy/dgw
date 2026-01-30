#include <catch2/catch_test_macros.hpp>
#include "dgw/dgw.hpp"

using namespace dgw;

TEST_CASE("Boussinesq solver creation", "[boussinesq]") {
    auto physics = create_physics(GoverningEquation::Boussinesq);
    REQUIRE(physics != nullptr);
    REQUIRE(physics->type() == GoverningEquation::Boussinesq);
    REQUIRE(physics->name() == "Boussinesq");
}
