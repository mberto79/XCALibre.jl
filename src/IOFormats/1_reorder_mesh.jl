export reorder_mesh!

"""
    reorder_mesh!(mesh::Union{Mesh2,Mesh3}; method=:rcm, polymesh="constant/polyMesh") -> mesh

Reorder the cells, faces and nodes of a host `mesh` in place so that neighbouring cells sit close in
memory. Cell loops read the values of each cell's neighbours, so meshes whose generator numbers cells
without regard to adjacency (typical of unstructured tetrahedral meshes) run faster once reordered.
Geometry, face orientation and boundary patches are unchanged; only the numbering differs. The extra
memory is a few index vectors of the mesh's integer type and one buffer the size of the largest
connectivity list.

- `method=:rcm`: reverse Cuthill-McKee ordering of the cell graph (bounded index distance between
  neighbours).
- `method=:morton`: Z-order curve of the cell centres (compact groups of consecutive cells).

Internal faces follow their lowest-numbered cell, boundary faces their owner cell within each patch,
and nodes the order in which the cells first reach them. The mesh is left unchanged unless the new
order shortens the mean index distance between neighbouring cells by at least 10%.

Fields built on the reordered mesh, and results written from them, are in the new order. So that the
mesh stored on disk matches, a 3D mesh in `polymesh` with the same number of points and faces but
numbered differently is rewritten in the new order (zone files there, which list cells by number, are
reported as no longer valid); pass `polymesh=nothing` to leave the files untouched. Serial meshes
only; call before `adapt`, and before `distribute` to reorder a mesh for a distributed run.

# Example

```julia
mesh = reorder_mesh!(UNV3D_mesh("mesh.unv", scale=0.001))
```
"""
function reorder_mesh!(mesh::Union{Mesh2,Mesh3}; method::Symbol=:rcm, polymesh="constant/polyMesh")
    _reorder_mesh!(mesh, method) === nothing && return mesh
    isnothing(polymesh) || _sync_polyMesh_order(mesh; dir=polymesh)
    mesh
end
