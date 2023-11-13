export write_vtk, model2vtk

function model2vtk(model::RANS{Laminar,F1,F2,V,T,E,D}, name) where {F1,F2,V,T,E,D}
    args = (
        ("U", model.U), 
        ("p", model.p)
    )
    write_vtk(name, model.mesh, args...)
end

function model2vtk(model::RANS{KOmega,F1,F2,V,T,E,D}, name) where {F1,F2,V,T,E,D}
    args = (
        ("U", model.U), 
        ("p", model.p),
        ("k", model.turbulence.k),
        ("omega", model.turbulence.omega),
        ("nut", model.turbulence.nut)
    )
    write_vtk(name, model.mesh, args...)
end

function write_vtk_3D(name,mesh,args...)
    filename=name*".vtk"
    open(filename,"w") do io
        write(io,"# vtk DataFile Version 3.0\n")
        write(io,"Julia 3D CFD Simulation Data\n")
        write(io,"ASCII\n")
        write(io,"DATASET UNSTRUCTURED_GRID\n")
        nPoints=length(mesh.nodes)
        nCells=length(mesh.cells)
        write(io,"POINTS $(nPoints) float\n")
        for node ∈ mesh.nodes
            (;coords)=node
            println(io, coords[1]," ",coords[2]," ", coords[3])
        end

        sum=0

        for cell ∈ mesh.cells
            sum += length(cell.nodesID)
        end

        cellListSize=sum+nCells
        write(io,"CELLS $(nCells) $(cellListSize)\n")

        for cell ∈ mesh.cells
            nNodes=length(cell.nodesID)
            nodes=""
            for nodeID ∈ cell.nodesID
                node="$(nodeID-1)"
                nodes=nodes*" "*node
            end
            println(io,nNodes," ",nodes)
        end

        write(io,"CELL_TYPES $(nCells)\n")

        for cell ∈ mesh.cells
            nCellIDs = length(cell.nodesID)
            if nCellIDs == 3
                type = "5"
            elseif nCellIDs == 4
                type = "9"
            elseif nCellIDs > 4
                type = "7"
            end
            println(io, type)
        end

        write(io, "CELL_DATA $(nCells)\n")

        for arg ∈ args
            label = arg[1]
            field = arg[2]
            field_type = typeof(field)
            if field_type <: ScalarField
                write(io, "SCALARS $(label) double 1\n")
                write(io, "LOOKUP_TABLE CellColors\n")
                for value ∈ field.values
                    println(io, value)
                end
            elseif field_type <: VectorField
                write(io, "VECTORS $(label) double\n")
                for i ∈ eachindex(field.x)
                    println(io, field.x[i]," ",field.y[i] ," ",field.z[i] )
                end
            else
                throw("""
                Input data should be a ScalarField or VectorField e.g. ("U", U)
                """)
            end
        end
    end





end