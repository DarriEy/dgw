/**
 * @file mesh.cpp
 * @brief Mesh implementation
 */

#include "dgw/core/mesh.hpp"
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <cmath>

namespace dgw {

// ============================================================================
// Mesh2D Implementation
// ============================================================================

std::vector<Index> Mesh2D::boundary_faces() const {
    std::vector<Index> result;
    for (Index f = 0; f < n_faces(); ++f) {
        if (is_boundary_face(f)) {
            result.push_back(f);
        }
    }
    return result;
}

std::vector<Index> Mesh2D::river_cells() const {
    std::vector<Index> result;
    for (const auto& [cell_id, seg_id] : cell_to_river_) {
        result.push_back(cell_id);
    }
    return result;
}

bool Mesh2D::has_river(Index cell_id) const {
    return cell_to_river_.find(cell_id) != cell_to_river_.end();
}

const RiverSegment* Mesh2D::river_segment(Index cell_id) const {
    auto it = cell_to_river_.find(cell_id);
    if (it == cell_to_river_.end()) {
        return nullptr;
    }
    return &river_segments_[it->second];
}

Real Mesh2D::intercell_distance(Index i, Index j) const {
    // Find face between cells i and j
    Index i_min = std::min(i, j);
    Index j_max = std::max(i, j);
    auto it = cell_pair_to_face_.find({i_min, j_max});
    if (it != cell_pair_to_face_.end()) {
        return faces_[it->second].distance;
    }
    // Not neighbors - compute from centroids
    Vec3 ci = cells_[i].centroid;
    Vec3 cj = cells_[j].centroid;
    return std::sqrt((ci.x() - cj.x()) * (ci.x() - cj.x()) +
                     (ci.y() - cj.y()) * (ci.y() - cj.y()));
}

Real Mesh2D::face_area_between(Index i, Index j) const {
    Index i_min = std::min(i, j);
    Index j_max = std::max(i, j);
    auto it = cell_pair_to_face_.find({i_min, j_max});
    if (it != cell_pair_to_face_.end()) {
        return faces_[it->second].area;
    }
    return 0.0;
}

void Mesh2D::add_node(Real x, Real y, Real z) {
    node_x_.push_back(x);
    node_y_.push_back(y);
    node_z_.push_back(z);
}

void Mesh2D::add_cell(const Cell& cell) {
    cells_.push_back(cell);
}

void Mesh2D::add_face(const Face& face) {
    faces_.push_back(face);
}

void Mesh2D::add_river_segment(const RiverSegment& seg) {
    Index seg_id = static_cast<Index>(river_segments_.size());
    river_segments_.push_back(seg);
    cell_to_river_[seg.cell_id] = seg_id;
    
    // Mark cell as river cell
    if (seg.cell_id >= 0 && seg.cell_id < static_cast<Index>(cells_.size())) {
        cells_[seg.cell_id].is_river_cell = true;
        cells_[seg.cell_id].river_segment_id = seg_id;
    }
}

void Mesh2D::finalize() {
    // Build cell_pair_to_face map
    cell_pair_to_face_.clear();
    for (Index f = 0; f < n_faces(); ++f) {
        Index i = faces_[f].cell_left;
        Index j = faces_[f].cell_right;
        if (i >= 0 && j >= 0) {
            Index i_min = std::min(i, j);
            Index j_max = std::max(i, j);
            cell_pair_to_face_[{i_min, j_max}] = f;
        }
    }
    
    // Compute bounds
    min_coords_ = Vec3(std::numeric_limits<Real>::max(),
                       std::numeric_limits<Real>::max(),
                       std::numeric_limits<Real>::max());
    max_coords_ = Vec3(std::numeric_limits<Real>::lowest(),
                       std::numeric_limits<Real>::lowest(),
                       std::numeric_limits<Real>::lowest());
    
    for (const auto& cell : cells_) {
        min_coords_.x() = std::min(min_coords_.x(), cell.centroid.x());
        min_coords_.y() = std::min(min_coords_.y(), cell.centroid.y());
        min_coords_.z() = std::min(min_coords_.z(), cell.z_bottom);
        max_coords_.x() = std::max(max_coords_.x(), cell.centroid.x());
        max_coords_.y() = std::max(max_coords_.y(), cell.centroid.y());
        max_coords_.z() = std::max(max_coords_.z(), cell.z_surface);
    }
}

void Mesh2D::set_hru_mapping(const SparseMatrix& hru_to_cell) {
    hru_to_cell_ = hru_to_cell;
}

// ============================================================================
// Mesh3D Implementation
// ============================================================================

std::vector<Index> Mesh3D::boundary_faces() const {
    std::vector<Index> result;
    for (Index f = 0; f < n_faces(); ++f) {
        if (is_boundary_face(f)) {
            result.push_back(f);
        }
    }
    return result;
}

Real Mesh3D::intercell_distance(Index i, Index j) const {
    Vec3 ci = cells_[i].centroid;
    Vec3 cj = cells_[j].centroid;
    return (ci - cj).norm();
}

Real Mesh3D::face_area_between(Index i, Index j) const {
    // Search through cell i's faces
    for (Index f : cells_[i].faces) {
        if ((faces_[f].cell_left == i && faces_[f].cell_right == j) ||
            (faces_[f].cell_left == j && faces_[f].cell_right == i)) {
            return faces_[f].area;
        }
    }
    return 0.0;
}

std::vector<Index> Mesh3D::surface_cells() const {
    std::vector<Index> result;
    // Surface cells have a face with no upper neighbor (boundary at top)
    for (Index i = 0; i < n_cells(); ++i) {
        for (Index f : cells_[i].faces) {
            if (is_boundary_face(f)) {
                const Face& face = faces_[f];
                // Check if face normal points upward (top boundary)
                if (face.normal.z() > 0.5) {
                    result.push_back(i);
                    break;
                }
            }
        }
    }
    return result;
}

std::vector<Index> Mesh3D::bottom_cells() const {
    std::vector<Index> result;
    for (Index i = 0; i < n_cells(); ++i) {
        for (Index f : cells_[i].faces) {
            if (is_boundary_face(f)) {
                const Face& face = faces_[f];
                // Check if face normal points downward (bottom boundary)
                if (face.normal.z() < -0.5) {
                    result.push_back(i);
                    break;
                }
            }
        }
    }
    return result;
}

void Mesh3D::add_node(const Vec3& coords) {
    node_coords_.push_back(coords);
}

void Mesh3D::add_cell(const Cell& cell, Index layer) {
    cells_.push_back(cell);
    cell_layers_.push_back(layer);
}

void Mesh3D::add_face(const Face& face) {
    faces_.push_back(face);
}

void Mesh3D::finalize() {
    // Compute bounds
    min_coords_ = Vec3(std::numeric_limits<Real>::max(),
                       std::numeric_limits<Real>::max(),
                       std::numeric_limits<Real>::max());
    max_coords_ = Vec3(std::numeric_limits<Real>::lowest(),
                       std::numeric_limits<Real>::lowest(),
                       std::numeric_limits<Real>::lowest());
    
    for (const auto& cell : cells_) {
        min_coords_ = min_coords_.cwiseMin(cell.centroid);
        max_coords_ = max_coords_.cwiseMax(cell.centroid);
    }
}

// ============================================================================
// MeshLayered Implementation
// ============================================================================

MeshLayered::MeshLayered(Ptr<Mesh2D> base_mesh, const std::vector<Layer>& layers)
    : base_mesh_(std::move(base_mesh))
    , layers_(layers)
{
    build_3d_cells();
    build_3d_faces();
    
    // Compute bounds
    min_coords_ = base_mesh_->min_coords();
    max_coords_ = base_mesh_->max_coords();
    if (!layers_.empty()) {
        min_coords_.z() = layers_.back().z_bottom;
        max_coords_.z() = layers_.front().z_top;
    }
}

Real MeshLayered::cell_volume(Index i) const {
    auto [col, layer] = cell_column_layer(i);
    Real area = base_mesh_->cell_volume(col);  // 2D area
    Real thickness = layers_[layer].z_top - layers_[layer].z_bottom;
    return area * thickness;
}

Vec3 MeshLayered::cell_centroid(Index i) const {
    auto [col, layer] = cell_column_layer(i);
    Vec3 base_centroid = base_mesh_->cell_centroid(col);
    Real z_mid = 0.5 * (layers_[layer].z_top + layers_[layer].z_bottom);
    return Vec3(base_centroid.x(), base_centroid.y(), z_mid);
}

std::span<const Index> MeshLayered::cell_neighbors(Index i) const {
    return std::span<const Index>(cells_3d_[i].neighbors);
}

std::span<const Index> MeshLayered::cell_faces(Index i) const {
    return std::span<const Index>(cells_3d_[i].faces);
}

bool MeshLayered::is_boundary_face(Index f) const {
    return faces_3d_[f].cell_left < 0 || faces_3d_[f].cell_right < 0;
}

std::vector<Index> MeshLayered::boundary_faces() const {
    std::vector<Index> result;
    for (Index f = 0; f < n_faces(); ++f) {
        if (is_boundary_face(f)) {
            result.push_back(f);
        }
    }
    return result;
}

std::vector<Index> MeshLayered::river_cells() const {
    // River cells are in top layer only
    std::vector<Index> base_river = base_mesh_->river_cells();
    std::vector<Index> result;
    result.reserve(base_river.size());
    for (Index col : base_river) {
        result.push_back(cell_index(col, 0));  // Layer 0 = top
    }
    return result;
}

bool MeshLayered::has_river(Index cell_id) const {
    auto [col, layer] = cell_column_layer(cell_id);
    return layer == 0 && base_mesh_->has_river(col);
}

const RiverSegment* MeshLayered::river_segment(Index cell_id) const {
    auto [col, layer] = cell_column_layer(cell_id);
    if (layer != 0) return nullptr;
    return base_mesh_->river_segment(col);
}

Real MeshLayered::intercell_distance(Index i, Index j) const {
    auto [col_i, layer_i] = cell_column_layer(i);
    auto [col_j, layer_j] = cell_column_layer(j);
    
    if (col_i == col_j) {
        // Same column, vertical distance
        Real z_i = 0.5 * (layers_[layer_i].z_top + layers_[layer_i].z_bottom);
        Real z_j = 0.5 * (layers_[layer_j].z_top + layers_[layer_j].z_bottom);
        return std::abs(z_i - z_j);
    } else if (layer_i == layer_j) {
        // Same layer, horizontal distance
        return base_mesh_->intercell_distance(col_i, col_j);
    } else {
        // Diagonal (shouldn't happen in standard stencil)
        Vec3 ci = cell_centroid(i);
        Vec3 cj = cell_centroid(j);
        return (ci - cj).norm();
    }
}

Real MeshLayered::face_area_between(Index i, Index j) const {
    auto [col_i, layer_i] = cell_column_layer(i);
    auto [col_j, layer_j] = cell_column_layer(j);
    
    if (col_i == col_j) {
        // Vertical face (horizontal area)
        return base_mesh_->cell_volume(col_i);  // 2D area
    } else {
        // Horizontal face (vertical area = edge_length * thickness)
        Real edge_length = base_mesh_->face_area_between(col_i, col_j);
        Real thickness = layers_[layer_i].z_top - layers_[layer_i].z_bottom;
        return edge_length * thickness;
    }
}

std::vector<Index> MeshLayered::column_cells(Index col) const {
    std::vector<Index> result;
    result.reserve(n_layers());
    for (Index layer = 0; layer < n_layers(); ++layer) {
        result.push_back(cell_index(col, layer));
    }
    return result;
}

std::vector<Index> MeshLayered::layer_cells(Index layer) const {
    std::vector<Index> result;
    result.reserve(base_mesh_->n_cells());
    for (Index col = 0; col < base_mesh_->n_cells(); ++col) {
        result.push_back(cell_index(col, layer));
    }
    return result;
}

std::vector<Index> MeshLayered::surface_cells() const {
    return layer_cells(0);
}

std::vector<Index> MeshLayered::bottom_cells() const {
    return layer_cells(n_layers() - 1);
}

void MeshLayered::build_3d_cells() {
    Index n_col = base_mesh_->n_cells();
    Index n_lay = static_cast<Index>(layers_.size());
    n_cells_ = n_col * n_lay;
    
    cells_3d_.resize(n_cells_);
    
    for (Index col = 0; col < n_col; ++col) {
        const Cell& base_cell = base_mesh_->cell(col);
        
        for (Index layer = 0; layer < n_lay; ++layer) {
            Index idx = cell_index(col, layer);
            Cell& cell3d = cells_3d_[idx];
            
            cell3d.id = idx;
            cell3d.centroid = cell_centroid(idx);
            cell3d.volume = cell_volume(idx);
            cell3d.z_surface = layers_[layer].z_top;
            cell3d.z_bottom = layers_[layer].z_bottom;
            cell3d.is_river_cell = (layer == 0) && base_cell.is_river_cell;
            cell3d.river_segment_id = cell3d.is_river_cell ? base_cell.river_segment_id : -1;
            
            // Build neighbors
            cell3d.neighbors.clear();
            
            // Horizontal neighbors (same layer)
            for (Index nb_col : base_cell.neighbors) {
                cell3d.neighbors.push_back(cell_index(nb_col, layer));
            }
            
            // Vertical neighbors
            if (layer > 0) {
                cell3d.neighbors.push_back(cell_index(col, layer - 1));  // Up
            }
            if (layer < n_lay - 1) {
                cell3d.neighbors.push_back(cell_index(col, layer + 1));  // Down
            }
        }
    }
}

void MeshLayered::build_3d_faces() {
    // Count faces: horizontal faces from base mesh * n_layers
    //            + vertical faces between layers * n_columns
    Index n_col = base_mesh_->n_cells();
    Index n_lay = static_cast<Index>(layers_.size());
    Index n_base_faces = base_mesh_->n_faces();
    
    n_faces_ = n_base_faces * n_lay + n_col * (n_lay - 1);
    faces_3d_.clear();
    faces_3d_.reserve(n_faces_);
    
    Index face_id = 0;
    
    // Horizontal faces (between columns in same layer)
    for (Index layer = 0; layer < n_lay; ++layer) {
        for (Index bf = 0; bf < n_base_faces; ++bf) {
            const Face& base_face = base_mesh_->face(bf);
            
            Face face3d;
            face3d.id = face_id++;
            face3d.cell_left = base_face.cell_left >= 0 ? 
                              cell_index(base_face.cell_left, layer) : -1;
            face3d.cell_right = base_face.cell_right >= 0 ?
                               cell_index(base_face.cell_right, layer) : -1;
            face3d.area = base_face.area * (layers_[layer].z_top - layers_[layer].z_bottom);
            face3d.distance = base_face.distance;
            face3d.normal = Vec3(base_face.normal.x(), base_face.normal.y(), 0.0);
            face3d.centroid = Vec3(base_face.centroid.x(), base_face.centroid.y(),
                                   0.5 * (layers_[layer].z_top + layers_[layer].z_bottom));
            face3d.bc_type = base_face.bc_type;
            face3d.bc_value = base_face.bc_value;
            
            faces_3d_.push_back(face3d);
        }
    }
    
    // Vertical faces (between layers in same column)
    for (Index col = 0; col < n_col; ++col) {
        Real area = base_mesh_->cell_volume(col);  // Horizontal area
        Vec3 xy = base_mesh_->cell_centroid(col);
        
        for (Index layer = 0; layer < n_lay - 1; ++layer) {
            Face face3d;
            face3d.id = face_id++;
            face3d.cell_left = cell_index(col, layer);      // Upper cell
            face3d.cell_right = cell_index(col, layer + 1); // Lower cell
            face3d.area = area;
            face3d.distance = 0.5 * (layers_[layer].z_top - layers_[layer].z_bottom +
                                     layers_[layer+1].z_top - layers_[layer+1].z_bottom);
            face3d.normal = Vec3(0, 0, -1);  // Pointing downward
            face3d.centroid = Vec3(xy.x(), xy.y(), layers_[layer].z_bottom);
            face3d.bc_type = BoundaryType::NoFlow;  // Internal face
            
            faces_3d_.push_back(face3d);
        }
    }
    
    // Link faces to cells
    for (Index f = 0; f < n_faces(); ++f) {
        const Face& face = faces_3d_[f];
        if (face.cell_left >= 0) {
            cells_3d_[face.cell_left].faces.push_back(f);
        }
        if (face.cell_right >= 0) {
            cells_3d_[face.cell_right].faces.push_back(f);
        }
    }
}

// ============================================================================
// Factory Methods
// ============================================================================

Ptr<Mesh> Mesh::from_file(const std::string& filename) {
    // Determine format from extension
    std::string ext = filename.substr(filename.find_last_of('.') + 1);
    
    if (ext == "nc" || ext == "nc4") {
        return mesh_io::load_netcdf(filename);
    } else if (ext == "msh") {
        return mesh_io::load_gmsh(filename);
    } else {
        throw std::runtime_error("Unknown mesh file format: " + ext);
    }
}

Ptr<Mesh> Mesh::create_structured_2d(
    Index nx, Index ny, Real dx, Real dy,
    const Vector& z_surface, const Vector& z_bottom)
{
    auto mesh = std::make_shared<Mesh2D>();
    
    // Create cells
    for (Index j = 0; j < ny; ++j) {
        for (Index i = 0; i < nx; ++i) {
            Index idx = j * nx + i;
            
            Cell cell;
            cell.id = idx;
            cell.centroid = Vec3((i + 0.5) * dx, (j + 0.5) * dy, 0.0);
            cell.volume = dx * dy;
            cell.z_surface = z_surface(idx);
            cell.z_bottom = z_bottom(idx);
            cell.is_river_cell = false;
            cell.river_segment_id = -1;
            
            // Add neighbors
            if (i > 0) cell.neighbors.push_back(idx - 1);
            if (i < nx - 1) cell.neighbors.push_back(idx + 1);
            if (j > 0) cell.neighbors.push_back(idx - nx);
            if (j < ny - 1) cell.neighbors.push_back(idx + nx);
            
            mesh->add_cell(cell);
        }
    }
    
    // Create faces
    Index face_id = 0;
    
    // Vertical faces (between columns)
    for (Index j = 0; j < ny; ++j) {
        for (Index i = 0; i <= nx; ++i) {
            Face face;
            face.id = face_id++;
            face.cell_left = (i > 0) ? j * nx + (i - 1) : -1;
            face.cell_right = (i < nx) ? j * nx + i : -1;
            face.area = dy;
            face.distance = dx;
            face.normal = Vec3(1, 0, 0);
            face.centroid = Vec3(i * dx, (j + 0.5) * dy, 0);
            face.bc_type = BoundaryType::NoFlow;
            face.bc_value = 0.0;

            mesh->add_face(face);
        }
    }

    // Horizontal faces (between rows)
    for (Index j = 0; j <= ny; ++j) {
        for (Index i = 0; i < nx; ++i) {
            Face face;
            face.id = face_id++;
            face.cell_left = (j > 0) ? (j - 1) * nx + i : -1;
            face.cell_right = (j < ny) ? j * nx + i : -1;
            face.area = dx;
            face.distance = dy;
            face.normal = Vec3(0, 1, 0);
            face.centroid = Vec3((i + 0.5) * dx, j * dy, 0);
            face.bc_type = BoundaryType::NoFlow;
            face.bc_value = 0.0;
            
            mesh->add_face(face);
        }
    }
    
    // Populate cell face lists from face connectivity
    for (Index f = 0; f < static_cast<Index>(mesh->n_faces()); ++f) {
        const Face& face = mesh->face(f);
        if (face.cell_left >= 0) {
            mesh->cell_mut(face.cell_left).faces.push_back(f);
        }
        if (face.cell_right >= 0) {
            mesh->cell_mut(face.cell_right).faces.push_back(f);
        }
    }

    mesh->finalize();
    return mesh;
}

// ============================================================================
// mesh_io stubs (full implementations require NetCDF/Gmsh libraries)
// ============================================================================

namespace mesh_io {

Ptr<Mesh> load_netcdf(const std::string& filename) {
#ifdef DGW_HAS_NETCDF
    // TODO: implement NetCDF mesh loading
    throw std::runtime_error("NetCDF mesh loading not yet implemented: " + filename);
#else
    throw std::runtime_error("NetCDF support not compiled. Cannot load: " + filename);
#endif
}

Ptr<Mesh> load_gmsh(const std::string& filename) {
    throw std::runtime_error("Gmsh mesh loading not yet implemented: " + filename);
}

Ptr<Mesh> load_ugrid(const std::string& filename) {
    throw std::runtime_error("UGRID mesh loading not yet implemented: " + filename);
}

void save_netcdf(const Mesh& /*mesh*/, const std::string& filename) {
#ifdef DGW_HAS_NETCDF
    // TODO: implement NetCDF mesh saving
    throw std::runtime_error("NetCDF mesh saving not yet implemented: " + filename);
#else
    throw std::runtime_error("NetCDF support not compiled. Cannot save: " + filename);
#endif
}

void save_vtk(const Mesh& /*mesh*/, const std::string& filename) {
    throw std::runtime_error("VTK mesh saving not yet implemented: " + filename);
}

} // namespace mesh_io

} // namespace dgw
