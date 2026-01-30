#include <catch2/catch_test_macros.hpp>
#include "dgw/dgw.hpp"

using namespace dgw;

TEST_CASE("LinearDiffusion steady state", "[linear_diffusion]") {
    PhysicsDecisions decisions;
    decisions.governing_equation = GoverningEquation::LinearDiffusion;

    auto physics = create_physics(GoverningEquation::LinearDiffusion);
    REQUIRE(physics != nullptr);
    REQUIRE(physics->type() == GoverningEquation::LinearDiffusion);
}
