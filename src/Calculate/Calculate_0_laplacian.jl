export laplacian!

# Warning Laplacian Solver Only Works For Orthogonal Grids
function laplacian!(phi_out, phif_in ,phi_in, BCs, time, config; disp_warn=true)
    if disp_warn
        @warn "laplacian function currently does not support non-orthogonal or non grid meshes - This Should Be Fixed"
    end
    interpolate!(phif_in, phi_in, config)
    correct_boundaries!(phif_in, phi_in, BCs, time, config)
    mesh = phi_out.mesh

    
    vec_field = VectorField(mesh)
    initialise!(vec_field, [0,0,0])
    for (i,boundary) ∈ enumerate(mesh.boundaries)
        (; IDs_range) = boundary
        bcellID = mesh.boundary_cellsID[IDs_range]
        for i ∈ eachindex(IDs_range)
            fcentre = mesh.faces[IDs_range[i]].centre
            ccentre = mesh.cells[bcellID[i]].centre
            Δpos = 2 .*(fcentre.-ccentre)
            if abs(Δpos[1]) > 1e-10
                vec_field.x.values[bcellID[i]] += 2*abs(mesh.faces[IDs_range[i]].normal[1])*(phif_in[IDs_range[i]]-phi_in[bcellID[i]])/abs2(Δpos[1])
            end
            if abs(Δpos[2]) > 1e-10
                vec_field.y.values[bcellID[i]] += 2*abs(mesh.faces[IDs_range[i]].normal[2])*(phif_in[IDs_range[i]]-phi_in[bcellID[i]])/abs2(Δpos[2])
            end
            if abs(Δpos[3]) > 1e-10
                vec_field.z.values[bcellID[i]] += 2*abs(mesh.faces[IDs_range[i]].normal[3])*(phif_in[IDs_range[i]]-phi_in[bcellID[i]])/abs2(Δpos[3])
            end
            #println("$(fcentre), $(ccentre), $(Δpos), $(phif_in[IDs_range[i]]), $(phi_in[bcellID[i]]), $(vec_field.x.values[bcellID[i]])")
        end
    end
    for i ∈ eachindex(mesh.cells)
        (; faces_range, centre) = mesh.cells[i]
        main_centre = centre

        for fID ∈ mesh.cell_faces[faces_range]
            (; normal, ownerCells, delta) = mesh.faces[fID]
            for cellID ∈ ownerCells
                (; centre) = mesh.cells[cellID]
                Δpos = centre.-main_centre

                if abs(Δpos[1]) > 1e-10
                    vec_field.x[i] += abs(mesh.faces[fID].normal[1])*(phi_in[cellID]-phi_in[i])/abs2(Δpos[1])
                end
                if abs(Δpos[2]) > 1e-10
                    vec_field.y[i] += abs(mesh.faces[fID].normal[2])*(phi_in[cellID]-phi_in[i])/abs2(Δpos[2])
                end
                if abs(Δpos[3]) > 1e-10
                    vec_field.z[i] += abs(mesh.faces[fID].normal[3])*(phi_in[cellID]-phi_in[i])/abs2(Δpos[3])
                end
            end
        end

    end
    @. phi_out.values = vec_field.x.values + vec_field.y.values + vec_field.z.values

end