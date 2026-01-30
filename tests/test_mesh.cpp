#include <catch2/catch_test_macros.hpp>
#include "dgw/core/mesh.hpp"

using namespace dgw;

TEST_CASE("Mesh2D creation", "[mesh]") {
    auto mesh = Mesh::create_structured_2d(
        3, 3, 100.0, 100.0,
        Vector::Constant(9, 10.0),
        Vector::Constant(9, 0.0)
    );

    REQUIRE(mesh != nullptr);
    REQUIRE(mesh->n_cells() == 9);
    REQUIRE(mesh->dimension() == 2);
}

TEST_CASE("Mesh boundary faces", "[mesh]") {
    auto mesh = Mesh::create_structured_2d(
        3, 3, 100.0, 100.0,
        Vector::Constant(9, 10.0),
        Vector::Constant(9, 0.0)
    );

    auto bf = mesh->boundary_faces();
    REQUIRE(bf.size() > 0);
}
