cmake_minimum_required(VERSION 3.20)
project(dgw 
    VERSION 0.1.0
    DESCRIPTION "Differentiable Groundwater Model with Enzyme AD"
    LANGUAGES CXX
)

# ============================================================================
# Build Options
# ============================================================================
option(DGW_ENABLE_ENZYME "Enable Enzyme automatic differentiation" ON)
option(DGW_ENABLE_PYTHON "Build Python bindings" ON)
option(DGW_ENABLE_PETSC "Enable PETSc for large-scale solvers" OFF)
option(DGW_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(DGW_ENABLE_TESTS "Build tests" ON)
option(DGW_ENABLE_EXAMPLES "Build examples" ON)
option(DGW_BUILD_SHARED "Build shared library" ON)

# ============================================================================
# C++ Standard and Compiler Settings
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -ffast-math)
    endif()
endif()

# ============================================================================
# Dependencies
# ============================================================================

# Eigen3 (required) - header-only linear algebra
find_package(Eigen3 REQUIRED NO_MODULE)

# OpenMP (optional)
if(DGW_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    endif()
endif()

# Enzyme (optional but recommended)
if(DGW_ENABLE_ENZYME)
    find_package(Enzyme QUIET)
    if(Enzyme_FOUND)
        message(STATUS "Enzyme found")
        add_definitions(-DDGW_HAS_ENZYME)
    else()
        message(WARNING "Enzyme not found. AD will use finite differences fallback.")
        set(DGW_ENABLE_ENZYME OFF)
    endif()
endif()

# PETSc (optional, for large-scale)
if(DGW_ENABLE_PETSC)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PETSC PETSc)
    if(PETSC_FOUND)
        message(STATUS "PETSc found: ${PETSC_VERSION}")
        add_definitions(-DDGW_HAS_PETSC)
    else()
        message(WARNING "PETSc not found. Using Eigen solvers only.")
        set(DGW_ENABLE_PETSC OFF)
    endif()
endif()

# Python bindings
if(DGW_ENABLE_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, fetching...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# NetCDF for I/O (optional)
find_package(netCDF QUIET)
if(netCDF_FOUND)
    message(STATUS "NetCDF found")
    add_definitions(-DDGW_HAS_NETCDF)
endif()

# ============================================================================
# Library Sources
# ============================================================================
set(DGW_CORE_SOURCES
    src/core/mesh.cpp
    src/core/state.cpp
    src/core/parameters.cpp
    src/core/config.cpp
)

set(DGW_PHYSICS_SOURCES
    src/physics/linear_diffusion.cpp
    src/physics/boussinesq.cpp
    src/physics/confined.cpp
    src/physics/two_layer.cpp
    src/physics/richards_3d.cpp
    src/physics/water_retention.cpp
)

set(DGW_COUPLING_SOURCES
    src/coupling/stream_aquifer.cpp
    src/coupling/recharge.cpp
    src/coupling/remapper.cpp
    src/coupling/boundary_conditions.cpp
)

set(DGW_SOLVER_SOURCES
    src/solvers/newton.cpp
    src/solvers/linear_solver.cpp
    src/solvers/time_stepping.cpp
)

set(DGW_BMI_SOURCES
    src/bmi/dgw_bmi.cpp
)

set(DGW_ALL_SOURCES
    ${DGW_CORE_SOURCES}
    ${DGW_PHYSICS_SOURCES}
    ${DGW_COUPLING_SOURCES}
    ${DGW_SOLVER_SOURCES}
    ${DGW_BMI_SOURCES}
)

# ============================================================================
# Main Library Target
# ============================================================================
if(DGW_BUILD_SHARED)
    add_library(dgw SHARED ${DGW_ALL_SOURCES})
else()
    add_library(dgw STATIC ${DGW_ALL_SOURCES})
endif()

target_include_directories(dgw
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(dgw
    PUBLIC
        Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(dgw PUBLIC OpenMP::OpenMP_CXX)
endif()

if(DGW_ENABLE_PETSC AND PETSC_FOUND)
    target_include_directories(dgw PUBLIC ${PETSC_INCLUDE_DIRS})
    target_link_libraries(dgw PUBLIC ${PETSC_LIBRARIES})
endif()

if(netCDF_FOUND)
    target_link_libraries(dgw PUBLIC netCDF::netcdf)
endif()

# Enzyme compiler plugin
if(DGW_ENABLE_ENZYME AND Enzyme_FOUND)
    target_compile_options(dgw PRIVATE 
        "SHELL:-fplugin=$<TARGET_FILE:Enzyme::ClangEnzyme>"
    )
endif()

# ============================================================================
# BMI Shared Library (for NextGen)
# ============================================================================
add_library(dgw_bmi SHARED src/bmi/dgw_bmi.cpp)
target_link_libraries(dgw_bmi PRIVATE dgw)
target_include_directories(dgw_bmi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# ============================================================================
# Python Bindings
# ============================================================================
if(DGW_ENABLE_PYTHON)
    pybind11_add_module(dgw_py python/dgw/bindings.cpp)
    target_link_libraries(dgw_py PRIVATE dgw)
    
    # Install Python module to site-packages
    install(TARGETS dgw_py
        LIBRARY DESTINATION ${Python3_SITELIB}/dgw
    )
endif()

# ============================================================================
# Tests
# ============================================================================
if(DGW_ENABLE_TESTS)
    enable_testing()
    
    # Fetch Catch2 for testing
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)
    
    add_executable(dgw_tests
        tests/test_mesh.cpp
        tests/test_linear_diffusion.cpp
        tests/test_boussinesq.cpp
        tests/test_richards.cpp
        tests/test_gradients.cpp
        tests/test_two_layer.cpp
    )
    target_link_libraries(dgw_tests PRIVATE dgw Catch2::Catch2WithMain)
    
    include(CTest)
    include(Catch)
    catch_discover_tests(dgw_tests)
endif()

# ============================================================================
# Examples
# ============================================================================
if(DGW_ENABLE_EXAMPLES)
    add_executable(example_theis examples/theis_well.cpp)
    target_link_libraries(example_theis PRIVATE dgw)
    
    add_executable(example_coupled examples/coupled_summa.cpp)
    target_link_libraries(example_coupled PRIVATE dgw)
    
    add_executable(example_richards examples/richards_column.cpp)
    target_link_libraries(example_richards PRIVATE dgw)
endif()

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(TARGETS dgw dgw_bmi
    EXPORT dgwTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/dgw
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT dgwTargets
    FILE dgwTargets.cmake
    NAMESPACE dgw::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dgw
)

# ============================================================================
# Package Configuration
# ============================================================================
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dgwConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/dgwConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dgw
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/dgwConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/dgwConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/dgwConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dgw
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "dGW Configuration Summary")
message(STATUS "=========================")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Enzyme AD:      ${DGW_ENABLE_ENZYME}")
message(STATUS "  Python:         ${DGW_ENABLE_PYTHON}")
message(STATUS "  PETSc:          ${DGW_ENABLE_PETSC}")
message(STATUS "  OpenMP:         ${DGW_ENABLE_OPENMP}")
message(STATUS "  Tests:          ${DGW_ENABLE_TESTS}")
message(STATUS "")
