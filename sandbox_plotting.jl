using Plots
using FVM_1D
using Krylov

#mesh_file="src/UNV_3D/5_cell_new_boundaries.unv"
mesh_file="src/UNV_3D/5_cell_new_boundaries.unv"
mesh_file="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.06mm.unv"

mesh=build_mesh3D(mesh_file)

_x(node::Node) = node.coords[1]
_y(node::Node) = node.coords[2]
_z(node::Node) = node.coords[3]

_x(face::Face3D) = face.centre[1]
_y(face::Face3D) = face.centre[2]
_z(face::Face3D) = face.centre[3]

_x(cell::Cell) = cell.centre[1]
_y(cell::Cell) = cell.centre[2]
_z(cell::Cell) = cell.centre[3]

_coords(obj::Node) = [obj.coords[1]],[obj.coords[2]],[obj.coords[3]]
_centre(obj::Face3D) = [obj.centre[1]],[obj.centre[2]],[obj.centre[3]]
_centre(obj::Cell) = [obj.centre[1]],[obj.centre[2]],[obj.centre[3]]
_face_edges(obj::Vector{Node{A,B}}) where {A,B} = begin
    (
        [obj[i].coords[1] for i ∈ [1,2,3,1]],
        [obj[i].coords[2] for i ∈ [1,2,3,1]],
        [obj[i].coords[3] for i ∈ [1,2,3,1]]
    )
end

plot_nodes_IDs!(fig, mesh) = begin
    nodes = mesh.nodes
    for (ni, node) ∈ enumerate(nodes)
        annotate!(
            _x(node), _y(node), _z(node), 
            text(ni))
    end
    fig
end

plot_faces_IDs!(fig, mesh) = begin
    faces = mesh.faces
    for (fi, face) ∈ enumerate(faces)
        annotate!(
            _x(face), _y(face), _z(face), 
            text(fi, :blue))
    end
    fig
end

plot_cells_IDs!(fig, mesh) = begin
    cells = mesh.cells
    for (ci, cell) ∈ enumerate(cells)
        annotate!(
            _x(cell), _y(cell), _z(cell), 
            text(ci, :red))
    end
    fig
end

plot_face_edges!(fig, mesh) = begin
    (; nodes, faces, face_nodes) = mesh
    for (fi, face) ∈ enumerate(faces)
        nodesID = face_nodes[face.nodes_range]
        fnodes = nodes[nodesID]
        plot!(
            fig, _face_edges(fnodes), 
            label=false, color=:black,
            alpha=0.2)
    end
    fig
end

plot_cell_edges!(fig, mesh, celli) = begin
    (; nodes, cells, faces, face_nodes) = mesh
    for fi ∈ cells[celli].faces_range
        nodesID = face_nodes[faces[fi].nodes_range]
        fnodes = nodes[nodesID]
        println(fnodes)
        plot!(
            fig, _face_edges(fnodes), 
            label=false, color=:black,
            alpha=0.2)
    end
    fig
end




plotlyjs() # back end for an interactive plot
fig = scatter(_coords.(mesh.nodes), label=false, color=:black)
scatter!(_centre.(mesh.faces), label=false, color=:blue)
scatter!(_centre.(mesh.cells), label=false, color=:red)
plot_face_edges!(fig, mesh)
plot_cell_edges!(fig,mesh,5)

gr() # not for interactive inspection but fast for annimations
fig = scatter3d(xlabel="x", ylabel="y", zlabel="z")
plot_face_edges!(fig, mesh)
plot_nodes_IDs!(fig, mesh) # needs an existing plot
plot_faces_IDs!(fig, mesh) # needs an existing plot
plot_cells_IDs!(fig,mesh)
plot3d!(fig, camera=(45,20))

#plot_cell_edges!(fig, mesh, 1)

@gif for angle ∈ range(45, stop = 500, length = 1000)
    plot3d!(fig, camera=(angle,20))
end every 5
# Save the output in vscode (look for save icon where plots are shown)
