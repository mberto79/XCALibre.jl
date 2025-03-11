export generate!

function generate!(builder::MeshBuilder2D{I,F}) where {I,F}
    GC.gc()
    stats = @timed generate_mesh!(builder)
    mesh = stats.value
    println("Mesh information")
    println("")
    println("Number of cells: ", length(mesh.cells))
    println("Number of faces: ", length(mesh.faces))
    println("Number of nodes: ", length(mesh.nodes))
    println("Number of boundaries: ", length(mesh.boundaries))
    println("")
    println("Mesh built in ", stats.time," seconds (allocated ", stats.bytes," bytes)")
    mesh
end

function generate_mesh!(builder::MeshBuilder2D{I,F}) where {I,F}
    mesh = build!(builder)
    connect!(mesh, builder)
    geometry!(mesh)
    mesh
end