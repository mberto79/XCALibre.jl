include("mesh_reader_gambit.jl")
include("mesh_reader_gmsh.jl")

struct mesh_struct
    Nv::Int32
    nodes::Matrix{Float64}
    K::Int32
    EToV::Matrix{Int32}
    BCType::Matrix{Int32}
end

function mesh_reader(caseinfo)

    type = caseinfo["mesh_info"]["type"]
    fname = caseinfo["mesh_info"]["meshfile"]

    if type == "gmsh"
        mesh = mesh_reader_gmsh(fname)
    elseif type == "gambit"
        mesh = mesh_reader_gambit_2d(fname)
    end

    return mesh
end