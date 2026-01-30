/**
 * @file bmi.hpp
 * @brief Basic Model Interface (BMI) definitions
 * 
 * Implements the CSDMS Basic Model Interface standard for
 * coupling with NextGen and other BMI-compliant frameworks.
 * 
 * See: https://bmi.readthedocs.io/
 */

#pragma once

#include <string>
#include <vector>

namespace dgw {
namespace bmi {

/**
 * @brief Basic Model Interface abstract class
 * 
 * This is the standard BMI interface that NextGen expects.
 */
class Bmi {
public:
    virtual ~Bmi() = default;
    
    // ========================================================================
    // Model Control Functions
    // ========================================================================
    
    /**
     * @brief Initialize the model from a configuration file
     * @param config_file Path to configuration file
     */
    virtual void Initialize(std::string config_file) = 0;
    
    /**
     * @brief Advance model state by one time step
     */
    virtual void Update() = 0;
    
    /**
     * @brief Advance model state until the given time
     * @param time Target time
     */
    virtual void UpdateUntil(double time) = 0;
    
    /**
     * @brief Finalize the model, clean up resources
     */
    virtual void Finalize() = 0;
    
    // ========================================================================
    // Model Information Functions
    // ========================================================================
    
    /**
     * @brief Get the name of the model component
     */
    virtual std::string GetComponentName() = 0;
    
    /**
     * @brief Get the number of input variables
     */
    virtual int GetInputItemCount() = 0;
    
    /**
     * @brief Get the number of output variables
     */
    virtual int GetOutputItemCount() = 0;
    
    /**
     * @brief Get names of input variables
     */
    virtual std::vector<std::string> GetInputVarNames() = 0;
    
    /**
     * @brief Get names of output variables
     */
    virtual std::vector<std::string> GetOutputVarNames() = 0;
    
    // ========================================================================
    // Variable Information Functions
    // ========================================================================
    
    /**
     * @brief Get the grid identifier for a variable
     */
    virtual int GetVarGrid(std::string name) = 0;
    
    /**
     * @brief Get the data type of a variable
     */
    virtual std::string GetVarType(std::string name) = 0;
    
    /**
     * @brief Get the units of a variable
     */
    virtual std::string GetVarUnits(std::string name) = 0;
    
    /**
     * @brief Get the size (in bytes) of one element of a variable
     */
    virtual int GetVarItemsize(std::string name) = 0;
    
    /**
     * @brief Get the total size (in bytes) of a variable
     */
    virtual int GetVarNbytes(std::string name) = 0;
    
    /**
     * @brief Get the location of a variable on the grid
     * 
     * @return "node", "edge", or "face"
     */
    virtual std::string GetVarLocation(std::string name) = 0;
    
    // ========================================================================
    // Time Functions
    // ========================================================================
    
    /**
     * @brief Get the current model time
     */
    virtual double GetCurrentTime() = 0;
    
    /**
     * @brief Get the start time
     */
    virtual double GetStartTime() = 0;
    
    /**
     * @brief Get the end time
     */
    virtual double GetEndTime() = 0;
    
    /**
     * @brief Get the time step
     */
    virtual double GetTimeStep() = 0;
    
    /**
     * @brief Get the units of time
     */
    virtual std::string GetTimeUnits() = 0;
    
    // ========================================================================
    // Variable Getter/Setter Functions
    // ========================================================================
    
    /**
     * @brief Get a copy of variable values
     */
    virtual void GetValue(std::string name, void *dest) = 0;
    
    /**
     * @brief Get a reference to variable values (if possible)
     */
    virtual void *GetValuePtr(std::string name) = 0;
    
    /**
     * @brief Get values at specific indices
     */
    virtual void GetValueAtIndices(std::string name, void *dest, int *inds, int count) = 0;
    
    /**
     * @brief Set variable values
     */
    virtual void SetValue(std::string name, void *src) = 0;
    
    /**
     * @brief Set values at specific indices
     */
    virtual void SetValueAtIndices(std::string name, int *inds, int count, void *src) = 0;
    
    // ========================================================================
    // Grid Information Functions
    // ========================================================================
    
    /**
     * @brief Get the grid rank (number of dimensions)
     */
    virtual int GetGridRank(int grid) = 0;
    
    /**
     * @brief Get the total number of elements in the grid
     */
    virtual int GetGridSize(int grid) = 0;
    
    /**
     * @brief Get the grid type
     * 
     * @return "uniform_rectilinear", "rectilinear", "structured_quadrilateral",
     *         "unstructured", etc.
     */
    virtual std::string GetGridType(int grid) = 0;
    
    // For structured grids
    virtual void GetGridShape(int grid, int *shape) = 0;
    virtual void GetGridSpacing(int grid, double *spacing) = 0;
    virtual void GetGridOrigin(int grid, double *origin) = 0;
    
    // For rectilinear grids
    virtual void GetGridX(int grid, double *x) = 0;
    virtual void GetGridY(int grid, double *y) = 0;
    virtual void GetGridZ(int grid, double *z) = 0;
    
    // For unstructured grids
    virtual int GetGridNodeCount(int grid) = 0;
    virtual int GetGridEdgeCount(int grid) = 0;
    virtual int GetGridFaceCount(int grid) = 0;
    
    virtual void GetGridEdgeNodes(int grid, int *edge_nodes) = 0;
    virtual void GetGridFaceEdges(int grid, int *face_edges) = 0;
    virtual void GetGridFaceNodes(int grid, int *face_nodes) = 0;
    virtual void GetGridNodesPerFace(int grid, int *nodes_per_face) = 0;
};

} // namespace bmi
} // namespace dgw
