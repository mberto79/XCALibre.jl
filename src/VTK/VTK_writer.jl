export write_vtk

function write_vtk(mesh, scalarField) #, Ux, Uy, Uz, p)
    # UxNodes = FVM.NodeScalarField(Ux)
    # UyNodes = FVM.NodeScalarField(Uy)
    # UzNodes = FVM.NodeScalarField(Uz)
    # pNodes = FVM.NodeScalarField(p)
    # FVM.interpolate2nodes!(UxNodes, Ux)
    # FVM.interpolate2nodes!(UyNodes, Uy)
    # FVM.interpolate2nodes!(UzNodes, Uz)
    # FVM.interpolate2nodes!(pNodes, p)

    open("results.vtk", "w") do io
        write(io, "# vtk DataFile Version 3.0\n")
        write(io, "jCFD simulation data\n")
        write(io, "ASCII\n")
        write(io, "DATASET UNSTRUCTURED_GRID\n")
        nPoints = length(mesh.nodes)
        nCells = length(mesh.cells)
        write(io, "POINTS $(nPoints) double\n")
        for node ∈ mesh.nodes
            (; coords) = node
            println(io, coords[1]," ", coords[2]," ", coords[3])
        end
        sumIndeces = 0
        for cell ∈ mesh.cells
            sumIndeces += length(cell.nodesID)
        end
        cellListSize = sumIndeces + nCells
        write(io, "CELLS $(nCells) $(cellListSize)\n")
        for cell ∈ mesh.cells
            nNodes = length(cell.nodesID)
            nodes = ""
            for nodeID ∈ cell.nodesID 
                node = "$(nodeID-1)"
                nodes = nodes*" "*node
            end 
            println(io, nNodes," ", nodes)
        end
        write(io, "CELL_TYPES $(nCells)\n")
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
        write(io, "SCALARS phi float 1\n")
        write(io, "LOOKUP_TABLE CellColors\n")
        for value ∈ scalarField.values
            println(io, value)
        end
        # write(io, "POINT_DATA $(nPoints)\n")
        # write(io, "SCALARS p double 1\n")
        # write(io, "LOOKUP_TABLE default\n")
        # for p ∈ pNodes.values
        #     println(io, p)
        # end
        # write(io, "VECTORS U double\n")
        # for i ∈ 1:length(UxNodes.values)
        #     println(io, UxNodes.values[i]," ",UyNodes.values[i] ," ",UzNodes.values[i] )
        # end
        # # Boundary information
        # # to be implemented
    end
end