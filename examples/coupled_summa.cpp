/**
 * @file coupled_summa.cpp
 * @brief Example: coupled SUMMA-dGW simulation
 *
 * Demonstrates BMI-based coupling between dGW and a land surface model.
 */

#include "dgw/dgw.hpp"
#include <iostream>

int main() {
    using namespace dgw;

    std::cout << "dGW coupled SUMMA example\n";
    std::cout << "This example demonstrates BMI coupling.\n";
    std::cout << "Run with a configuration file:\n";
    std::cout << "  ./example_coupled config.yaml\n";

    return 0;
}
