using FVM_1D
using FVM_1D.FoamMesh



file_path = "unv_sample_meshes/OF_startrek/polyMesh/"
file_path = "unv_sample_meshes/OF_CRMHL_Wingbody_1v/polyMesh/"
integer=Int64
float=Float64
scale=0.001


points_file = joinpath(file_path,"points")
faces_file = joinpath(file_path,"faces")
neighbour_file = joinpath(file_path,"neighbour")
owner_file = joinpath(file_path,"owner")
boundary_file = joinpath(file_path,"boundary")

foamdata = FoamMesh.FoamMeshData(integer, float)

foamdata.points = FoamMesh.read_points(points_file, scale, integer, float)
foamdata.boundaries = FoamMesh.read_boundary(boundary_file, integer, float)

delimiters = ['(',' ', ')', '\n']

file_data = read(faces_file, String)
data_split = split(file_data, delimiters, keepempty=false)
data = tryparse.(Int64, data_split)
dataClean = filter(!isnothing, data)

# Find line with entry for total number of faces
startLine = 0
for (n, line) ∈ enumerate(eachline(faces_file)) 
    line_content = tryparse(Int64, line)
    if line_content !== nothing
        startLine = n
        println("Number of faces to read: $line_content (from line: $startLine)")
        break
    end
end

# Read file contents after header information
io = IOBuffer()
for (n, line) ∈ enumerate(eachline(faces_file)) 
    if n >= startLine
        println(io, line)
    end
end

file_data = String(take!(io))
data_split = split(file_data, delimiters, keepempty=false)
data = tryparse.(Int64, data_split)
dataClean = filter(!isnothing, data)

face_nodes = FoamMesh.read_faces(faces_file, integer, float)
face_neighbours = FoamMesh.read_neighbour(neighbour_file, integer, float)
face_owners = FoamMesh.read_owner(owner_file, integer, float)

FoamMesh.assign_faces!(foamdata, face_nodes, face_neighbours, face_owners, integer)