using FVM_1D
using StaticArrays
using LinearAlgebra

mesh_file = "unv_sample_meshes/quad.unv"
mesh_file = "unv_sample_meshes/nonorthogonal_45_degrees.unv"
mesh_file = "unv_sample_meshes/nonorthogonal_45_degrees_trig.unv"
mesh_file = "unv_sample_meshes/nonorthogonal_90_degrees.unv"

mesh = UNV2D_mesh(mesh_file, scale=0.001, integer_type=Int64, float_type=Float64)


linear_solution!(field) = begin
    mesh = field.mesh 
    values = field.values
    for (i, cell) ∈ enumerate(mesh.cells)
        point = cell.centre
        values[i] = 2*(point[1])
    end
end

config = (; hardware=(; backend=CPU(), workgroup=4))

F = ScalarField(mesh)
Ff = FaceScalarField(mesh)
∇F = Grad{Orthogonal}(F)
# ∇F = Grad{Midpoint}(F)

linear_solution!(F)

bnames = [:left, :right, :bottom, :top]

F = assign(F, Neumann.(bnames, Ref(0.0))...,)


grad!(∇F, Ff, F, F.BCs, config)

@time write_vtk("foamMeshTest", mesh, ("F", F), ("grad(F)", ∇F.result))


limit_gradient!(∇F, F)

@time write_vtk("foamMeshTest", mesh, ("F", F), ("grad(F)", ∇F.result))

diff = VectorField(mesh)

# Function defs
function limit_gradient!(∇F, F)
    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

    minPhi0 = maximum(F.values) # use min value so all values compared are larger
    maxPhi0 = minimum(F.values)

    for (cID, cell) ∈ enumerate(mesh.cells)
        minPhi = minPhi0 # reset for next cell
        maxPhi = maxPhi0

        # find min and max values around cell
        faces_range = cell.faces_range
        
        phiP = F[cID]
        # minPhi = phiP # reset for next cell
        # maxPhi = phiP
        for fi ∈ faces_range
            nID = cell_neighbours[fi]
            phiN = F[nID]
            maxPhi = max(phiN, maxPhi)
            minPhi = min(phiN, minPhi)
        end

        g0 = ∇F[cID]
        cc = cell.centre

        for fi ∈ faces_range 
            fID = cell_faces[fi]
            face = faces[fID]
            nID = face.ownerCells[2]
            phiN = F[nID]
            normal = face.normal
            nsign = cell_nsign[fi]
            na = nsign*normal

            fc = face.centre 
            cc_fc = fc - cc
            n0 = cc_fc/norm(cc_fc)
            gn = g0⋅n0
            δϕ = g0⋅cc_fc
            gτ = g0 - gn*n0
            if (maxPhi > phiP) && (δϕ > maxPhi - phiP)
                g0 = gτ + na*(maxPhi - phiP)
            elseif (minPhi < phiP) && (δϕ < minPhi - phiP)
                g0 = gτ + na*(minPhi - phiP)
            end            
        end
        ∇F.result.x.values[cID] = g0[1]
        ∇F.result.y.values[cID] = g0[2]
        ∇F.result.z.values[cID] = g0[3]
    end
end
