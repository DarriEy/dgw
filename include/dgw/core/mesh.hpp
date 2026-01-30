/**
 * @file mesh.hpp
 * @brief Unstructured mesh data structures for dGW
 * 
 * Supports 2D and 3D unstructured meshes with:
 * - Voronoi cells / triangles / tetrahedra
 * - River-conforming mesh edges
 * - Variable resolution
 * - Connectivity for finite volume discretization
 */

#pragma once

#include "types.hpp"
#include <span>
#include <unordered_map>

// Hash function for pair (must be defined before any unordered_map usage)
namespace dgw {
namespace detail {
struct PairHash {
    size_t operator()(const std::pair<Index, Index>& p) const {
        auto h1 = std::hash<Index>{}(p.first);
        auto h2 = std::hash<Index>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};
} // namespace detail
} // namespace dgw

namespace dgw {

// ============================================================================
// Face (Connection) Between Cells
// ============================================================================

/**
 * @brief Represents a face connecting two cells (or cell to boundary)
 */
struct Face {
    Index id;                   ///< Face index
    Index cell_left;            ///< Cell on left side (-1 if boundary)
    Index cell_right;           ///< Cell on right side (-1 if boundary)
    Real area;                  ///< Face area (3D) or length (2D) [m² or m]
    Real distance;              ///< Distance between cell centers [m]
    Vec3 normal;                ///< Unit normal (left → right)
    Vec3 centroid;              ///< Face centroid
    BoundaryType bc_type;       ///< Boundary type if at boundary
    Real bc_value;              ///< Boundary value if applicable
};

// ============================================================================
// Cell (Control Volume)
// ============================================================================

/**
 * @brief Represents a control volume cell
 */
struct Cell {
    Index id;                   ///< Cell index
    Vec3 centroid;              ///< Cell center coordinates
    Real volume;                ///< Cell volume [m³] (or area for 2D)
    Real z_surface;             ///< Surface elevation [m]
    Real z_bottom;              ///< Aquifer bottom elevation [m]
    std::vector<Index> faces;   ///< Indices of faces bounding this cell
    std::vector<Index> neighbors; ///< Indices of neighboring cells
    bool is_river_cell;         ///< Contains river segment
    Index river_segment_id;     ///< River segment index (-1 if none)
};

// ============================================================================
// River Segment
// ============================================================================

/**
 * @brief River segment for stream-aquifer coupling
 */
struct RiverSegment {
    Index id;                   ///< Segment index
    Index cell_id;              ///< Cell containing this segment
    Real length;                ///< Segment length [m]
    Real width;                 ///< Average width [m]
    Real streambed_elevation;   ///< Bottom of streambed [m]
    Real streambed_thickness;   ///< Clogging layer thickness [m]
    Real streambed_K;           ///< Streambed hydraulic conductivity [m/s]
    Index upstream_segment;     ///< Upstream segment (-1 if headwater)
    Index downstream_segment;   ///< Downstream segment (-1 if outlet)
};

// ============================================================================
// Layer Definition (for 3D layered meshes)
// ============================================================================

/**
 * @brief Vertical layer specification
 */
struct Layer {
    Index id;                   ///< Layer index (0 = top)
    Real z_top;                 ///< Layer top elevation [m]
    Real z_bottom;              ///< Layer bottom elevation [m]
    bool is_confined;           ///< True if confined aquifer
    std::string name;           ///< Layer name
};

// ============================================================================
// Base Mesh Class
// ============================================================================

/**
 * @brief Abstract base class for all mesh types
 */
class Mesh {
public:
    virtual ~Mesh() = default;
    
    // Basic info
    virtual MeshType type() const = 0;
    virtual Index n_cells() const = 0;
    virtual Index n_faces() const = 0;
    virtual Index n_nodes() const = 0;
    virtual Index n_layers() const { return 1; }
    virtual Index dimension() const = 0;
    
    // Cell access
    virtual const Cell& cell(Index i) const = 0;
    virtual Real cell_volume(Index i) const = 0;
    virtual Vec3 cell_centroid(Index i) const = 0;
    
    // Face access
    virtual const Face& face(Index f) const = 0;
    virtual std::pair<Index, Index> face_cells(Index f) const = 0;
    
    // Connectivity
    virtual std::span<const Index> cell_neighbors(Index i) const = 0;
    virtual std::span<const Index> cell_faces(Index i) const = 0;
    
    // Boundary
    virtual bool is_boundary_face(Index f) const = 0;
    virtual std::vector<Index> boundary_faces() const = 0;
    
    // River cells
    virtual std::vector<Index> river_cells() const = 0;
    virtual bool has_river(Index cell_id) const = 0;
    virtual const RiverSegment* river_segment(Index cell_id) const = 0;
    
    // Geometry helpers
    virtual Real intercell_distance(Index i, Index j) const = 0;
    virtual Real face_area_between(Index i, Index j) const = 0;
    
    // Coordinate bounds
    virtual Vec3 min_coords() const = 0;
    virtual Vec3 max_coords() const = 0;
    
    // Factory methods
    static Ptr<Mesh> from_file(const std::string& filename);
    static Ptr<Mesh> create_structured_2d(
        Index nx, Index ny, Real dx, Real dy,
        const Vector& z_surface, const Vector& z_bottom
    );
    static Ptr<Mesh> create_voronoi_2d(
        const std::vector<Vec2>& generators,
        const std::vector<std::array<Vec2, 2>>& boundary_segments,
        const Vector& z_surface, const Vector& z_bottom
    );
};

// ============================================================================
// 2D Unstructured Mesh
// ============================================================================

/**
 * @brief 2D unstructured mesh (Voronoi cells or triangles)
 */
class Mesh2D : public Mesh {
public:
    Mesh2D() = default;
    
    // Implement base class
    MeshType type() const override { return MeshType::Unstructured2D; }
    Index n_cells() const override { return static_cast<Index>(cells_.size()); }
    Index n_faces() const override { return static_cast<Index>(faces_.size()); }
    Index n_nodes() const override { return static_cast<Index>(node_x_.size()); }
    Index dimension() const override { return 2; }

    const Cell& cell(Index i) const override { return cells_[i]; }
    Cell& cell_mut(Index i) { return cells_[i]; }
    Real cell_volume(Index i) const override { return cells_[i].volume; }
    Vec3 cell_centroid(Index i) const override { return cells_[i].centroid; }
    
    const Face& face(Index f) const override { return faces_[f]; }
    std::pair<Index, Index> face_cells(Index f) const override {
        return {faces_[f].cell_left, faces_[f].cell_right};
    }
    
    std::span<const Index> cell_neighbors(Index i) const override {
        return std::span<const Index>(cells_[i].neighbors);
    }
    std::span<const Index> cell_faces(Index i) const override {
        return std::span<const Index>(cells_[i].faces);
    }
    
    bool is_boundary_face(Index f) const override {
        return faces_[f].cell_left < 0 || faces_[f].cell_right < 0;
    }
    
    std::vector<Index> boundary_faces() const override;
    std::vector<Index> river_cells() const override;
    bool has_river(Index cell_id) const override;
    const RiverSegment* river_segment(Index cell_id) const override;
    
    Real intercell_distance(Index i, Index j) const override;
    Real face_area_between(Index i, Index j) const override;
    
    Vec3 min_coords() const override { return min_coords_; }
    Vec3 max_coords() const override { return max_coords_; }
    
    // Mesh construction
    void add_node(Real x, Real y, Real z = 0.0);
    void add_cell(const Cell& cell);
    void add_face(const Face& face);
    void add_river_segment(const RiverSegment& seg);
    void finalize();  // Build connectivity, compute bounds
    
    // Remapping support
    void set_hru_mapping(const SparseMatrix& hru_to_cell);
    const SparseMatrix& hru_to_cell_map() const { return hru_to_cell_; }
    
private:
    // Node coordinates
    std::vector<Real> node_x_;
    std::vector<Real> node_y_;
    std::vector<Real> node_z_;
    
    // Cells and faces
    std::vector<Cell> cells_;
    std::vector<Face> faces_;
    
    // River segments
    std::vector<RiverSegment> river_segments_;
    std::unordered_map<Index, Index> cell_to_river_;  // cell_id → segment_id
    
    // Connectivity lookups
    std::unordered_map<std::pair<Index, Index>, Index,
        detail::PairHash> cell_pair_to_face_;
    
    // Remapping from SUMMA HRUs to GW cells
    SparseMatrix hru_to_cell_;
    
    // Bounds
    Vec3 min_coords_, max_coords_;
};

// ============================================================================
// 3D Unstructured Mesh
// ============================================================================

/**
 * @brief 3D unstructured mesh (tetrahedra or hexahedra)
 */
class Mesh3D : public Mesh {
public:
    Mesh3D() = default;
    
    MeshType type() const override { return MeshType::Unstructured3D; }
    Index n_cells() const override { return static_cast<Index>(cells_.size()); }
    Index n_faces() const override { return static_cast<Index>(faces_.size()); }
    Index n_nodes() const override { return static_cast<Index>(node_coords_.size()); }
    Index dimension() const override { return 3; }
    
    const Cell& cell(Index i) const override { return cells_[i]; }
    Real cell_volume(Index i) const override { return cells_[i].volume; }
    Vec3 cell_centroid(Index i) const override { return cells_[i].centroid; }
    
    const Face& face(Index f) const override { return faces_[f]; }
    std::pair<Index, Index> face_cells(Index f) const override {
        return {faces_[f].cell_left, faces_[f].cell_right};
    }
    
    std::span<const Index> cell_neighbors(Index i) const override {
        return std::span<const Index>(cells_[i].neighbors);
    }
    std::span<const Index> cell_faces(Index i) const override {
        return std::span<const Index>(cells_[i].faces);
    }
    
    bool is_boundary_face(Index f) const override {
        return faces_[f].cell_left < 0 || faces_[f].cell_right < 0;
    }
    
    std::vector<Index> boundary_faces() const override;
    std::vector<Index> river_cells() const override { return {}; }  // No 2D rivers in 3D mesh
    bool has_river(Index) const override { return false; }
    const RiverSegment* river_segment(Index) const override { return nullptr; }
    
    Real intercell_distance(Index i, Index j) const override;
    Real face_area_between(Index i, Index j) const override;
    
    Vec3 min_coords() const override { return min_coords_; }
    Vec3 max_coords() const override { return max_coords_; }
    
    // 3D-specific: vertical layers for Richards equation
    void set_layers(const std::vector<Layer>& layers) { layers_ = layers; }
    const std::vector<Layer>& layers() const { return layers_; }
    Index cell_layer(Index cell_id) const { return cell_layers_[cell_id]; }
    
    // Cells at surface (for recharge BC)
    std::vector<Index> surface_cells() const;
    
    // Cells at bottom (for deep BC)
    std::vector<Index> bottom_cells() const;
    
    void add_node(const Vec3& coords);
    void add_cell(const Cell& cell, Index layer = 0);
    void add_face(const Face& face);
    void finalize();
    
private:
    std::vector<Vec3> node_coords_;
    std::vector<Cell> cells_;
    std::vector<Face> faces_;
    std::vector<Layer> layers_;
    std::vector<Index> cell_layers_;  // cell_id → layer_id
    Vec3 min_coords_, max_coords_;
};

// ============================================================================
// Layered Mesh (2D extruded to 3D)
// ============================================================================

/**
 * @brief Layered mesh: 2D mesh extruded into vertical layers
 * 
 * Efficient for multi-layer aquifer systems where horizontal
 * resolution is independent of vertical structure.
 */
class MeshLayered : public Mesh {
public:
    MeshLayered(Ptr<Mesh2D> base_mesh, const std::vector<Layer>& layers);
    
    MeshType type() const override { return MeshType::Layered; }
    Index n_cells() const override { return n_cells_; }
    Index n_faces() const override { return n_faces_; }
    Index n_nodes() const override { return base_mesh_->n_nodes() * (layers_.size() + 1); }
    Index n_layers() const override { return static_cast<Index>(layers_.size()); }
    Index dimension() const override { return 3; }
    
    const Cell& cell(Index i) const override { return cells_3d_[i]; }
    Real cell_volume(Index i) const override;
    Vec3 cell_centroid(Index i) const override;
    
    const Face& face(Index f) const override { return faces_3d_[f]; }
    std::pair<Index, Index> face_cells(Index f) const override {
        return {faces_3d_[f].cell_left, faces_3d_[f].cell_right};
    }
    
    std::span<const Index> cell_neighbors(Index i) const override;
    std::span<const Index> cell_faces(Index i) const override;
    
    bool is_boundary_face(Index f) const override;
    std::vector<Index> boundary_faces() const override;
    std::vector<Index> river_cells() const override;
    bool has_river(Index cell_id) const override;
    const RiverSegment* river_segment(Index cell_id) const override;
    
    Real intercell_distance(Index i, Index j) const override;
    Real face_area_between(Index i, Index j) const override;
    
    Vec3 min_coords() const override { return min_coords_; }
    Vec3 max_coords() const override { return max_coords_; }
    
    // Layered-specific accessors
    const Mesh2D& base_mesh() const { return *base_mesh_; }
    const std::vector<Layer>& layers() const { return layers_; }
    
    /// Get 3D cell index from (2D column, layer)
    Index cell_index(Index col, Index layer) const {
        return col * n_layers() + layer;
    }
    
    /// Get (column, layer) from 3D cell index
    std::pair<Index, Index> cell_column_layer(Index cell_id) const {
        return {cell_id / n_layers(), cell_id % n_layers()};
    }
    
    /// Get all cells in a column
    std::vector<Index> column_cells(Index col) const;
    
    /// Get all cells in a layer
    std::vector<Index> layer_cells(Index layer) const;
    
    /// Get surface (top layer) cells
    std::vector<Index> surface_cells() const;
    
    /// Get bottom layer cells
    std::vector<Index> bottom_cells() const;
    
private:
    Ptr<Mesh2D> base_mesh_;
    std::vector<Layer> layers_;
    std::vector<Cell> cells_3d_;
    std::vector<Face> faces_3d_;
    Index n_cells_;
    Index n_faces_;
    Vec3 min_coords_, max_coords_;
    
    void build_3d_cells();
    void build_3d_faces();
};

// ============================================================================
// Mesh I/O
// ============================================================================

namespace mesh_io {

/// Load mesh from NetCDF file
Ptr<Mesh> load_netcdf(const std::string& filename);

/// Load mesh from Gmsh format
Ptr<Mesh> load_gmsh(const std::string& filename);

/// Load mesh from UGRID convention
Ptr<Mesh> load_ugrid(const std::string& filename);

/// Save mesh to NetCDF
void save_netcdf(const Mesh& mesh, const std::string& filename);

/// Save mesh to VTK for visualization
void save_vtk(const Mesh& mesh, const std::string& filename);

} // namespace mesh_io

// ============================================================================
// Mesh Generation Utilities
// ============================================================================

namespace mesh_gen {

/**
 * @brief Generate Voronoi mesh from point generators
 */
Ptr<Mesh2D> voronoi_from_points(
    const std::vector<Vec2>& generators,
    const std::vector<Vec2>& boundary_polygon,
    const Vector& z_surface,
    const Vector& z_bottom
);

/**
 * @brief Generate mesh with river-conforming edges
 */
Ptr<Mesh2D> river_conforming_mesh(
    const std::vector<Vec2>& boundary_polygon,
    const std::vector<std::vector<Vec2>>& river_polylines,
    Real base_resolution,
    Real river_resolution,
    const std::function<Real(Vec2)>& z_surface_func,
    const std::function<Real(Vec2)>& z_bottom_func
);

/**
 * @brief Refine mesh near wells
 */
void refine_near_wells(
    Mesh2D& mesh,
    const std::vector<Vec2>& well_locations,
    Real refinement_radius,
    Real target_resolution
);

} // namespace mesh_gen

} // namespace dgw

// PairHash is now defined in dgw::detail at the top of this file
