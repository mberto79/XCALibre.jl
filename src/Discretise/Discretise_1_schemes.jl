export schemes_and_sources!, sources!, set_b! #scheme!, scheme_source!

#= NOTE:
In source scheme the following indices are used and should be used with care:
cID - Index of the cell outer loop. Use to index "b" 
cIndex - Index of the cell based on sparse matrix. Use to index "nzval_array"
=#

# TIME 
## Steady
# @inline function scheme!(
#     term::Operator{F,P,I,Time{Steady}}, 
#     nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
#     nothing
# end
# @inline scheme_source!(
#     term::Operator{F,P,I,Time{Steady}}, 
#     b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
#     nothing
# end

# ## Euler
# @inline function scheme!(
#     term::Operator{F,P,I,Time{Euler}}, 
#     nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
#     nothing
# end
# @inline scheme_source!(
#     term::Operator{F,P,I,Time{Euler}}, 
#     b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
#         volume = cell.volume
#         rdt = 1/runtime.dt
#         nzval_array[cIndex] += volume*rdt
#         b[cID] += prev[cID]*volume*rdt
#     nothing
# end

# # LAPLACIAN

# @inline function scheme!(
#     term::Operator{F,P,I,Laplacian{Linear}}, 
#     nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
#     )  where {F,P,I}
#     ap = term.sign*(-term.flux[fID] * face.area)/face.delta
#     nzval_array[cIndex] += ap
#     nzval_array[nIndex] += -ap
#     nothing
# end
# @inline scheme_source!(
#     term::Operator{F,P,I,Laplacian{Linear}}, 
#     b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
#     nothing
# end

# # DIVERGENCE

# @inline function scheme!(
#     term::Operator{F,P,I,Divergence{Linear}}, 
#     nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
#     )  where {F,P,I}
#     xf = face.centre
#     xC = cell.centre
#     xN = cellN.centre
#     weight = norm(xf - xC)/norm(xN - xC)
#     one_minus_weight = one(eltype(weight)) - weight
#     ap = term.sign*(term.flux[fID]*ns)
#     nzval_array[cIndex] += ap*one_minus_weight
#     nzval_array[nIndex] += ap*weight
#     nothing
# end
# @inline scheme_source!(
#     term::Operator{F,P,I,Divergence{Linear}}, 
#     b, nzval_array, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
#     nothing
# end

# @inline function scheme!(
#     term::Operator{F,P,I,Divergence{Upwind}}, 
#     nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
#     )  where {F,P,I}
#     ap = term.sign*(term.flux[fID]*ns)
#     nzval_array[cIndex] += max(ap, 0.0)
#     nzval_array[nIndex] += -max(-ap, 0.0)
#     nothing
# end
# @inline scheme_source!(
#     term::Operator{F,P,I,Divergence{Upwind}}, 
#     b, nzval_array, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
#     nothing
# end

# # IMPLICIT SOURCE

# @inline function scheme!(
#     term::Operator{F,P,I,Si}, 
#     nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
#     )  where {F,P,I}
#     # ap = term.sign*(-term.flux[cIndex] * cell.volume)
#     # nzval_array[cIndex] += ap
#     nothing
# end
# @inline scheme_source!(
#     term::Operator{F,P,I,Si}, 
#     b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
#     phi = term.phi
#     # ap = max(flux, 0.0)
#     # ab = min(flux, 0.0)*phi[cID]
#     flux = term.sign*term.flux[cID]*cell.volume # indexed with cID
#     nzval_array[cIndex] += flux # indexed with cIndex
#     # flux = term.sign*term.flux[cID]*cell.volume*phi[cID]
#     # b[cID] -= flux
#     nothing
# end



@inline function schemes_and_sources!(
    term::Operator{F,P,I,Time{Steady}}, 
    nTerms, nSources, offset, fzero, ione, terms, rowval_array, colptr_array, nzval_array, cIndex, nIndex,  b, faces,
    cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
    backend, runtime, prev)  where {F,P,I}
    nothing
end

@inline function schemes_and_sources!(
    term::Operator{F,P,I,Time{Euler}}, 
    nTerms, nSources, offset, fzero, ione, terms, rowval_array, colptr_array, nzval_array, cIndex, nIndex,  b, faces,
    cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
    backend, runtime, prev)  where {F,P,I}

    kernel! = schemes_time_euler!(backend)
    kernel!(term, nTerms, nSources, offset, fzero, ione, terms, rowval_array, colptr_array, nzval_array, cIndex, nIndex,  b, faces,
            cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
            backend, runtime, prev, ndrange = length(cells))
    KernelAbstractions.synchronize(backend)
end

@kernel function schemes_time_euler!(term, nTerms, nSources, offset, fzero, ione, terms,  rowval_array, colptr_array, nzval_array, cIndex, nIndex, b, faces, cells, cell_faces, cell_neighbours, cell_nsign, integer, float, backend, runtime, prev)
    i = @index(Global)
    # (; terms) = model

    @inbounds begin
        # cell = cells[i]
        # @synchronize
        (; faces_range, volume) = cells[i]

        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            # face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]

            cIndex = nzval_index(colptr_array, rowval_array, i, i, ione)
            nIndex = nzval_index(colptr_array, rowval_array, nID, i, ione)

            # No scheme code for Euler time scheme

        end

    # scheme_scource loop
    # volume = cell.volume
    rdt = 1/runtime.dt
    Atomix.@atomic nzval_array[cIndex] += volume*rdt
    Atomix.@atomic b[i] += prev[i]*volume*rdt

    end
end

@inline function schemes_and_sources!(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    nTerms, nSources, offset, fzero, ione, terms, rowval_array, colptr_array, nzval_array, cIndex, nIndex,  b, faces,
    cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
    backend, runtime, prev)  where {F,P,I}

    kernel! = schemes_laplacian_linear!(backend)
    kernel!(term, nTerms, nSources, offset, fzero, ione, terms, rowval_array, colptr_array, nzval_array, cIndex, nIndex,  b, faces,
            cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
            backend, runtime, prev, ndrange = length(cells))
    KernelAbstractions.synchronize(backend)
end

@kernel function schemes_laplacian_linear!(term, nTerms, nSources, offset, fzero, ione, terms,  rowval_array, colptr_array, nzval_array, cIndex, nIndex, b, faces, cells, cell_faces, cell_neighbours, cell_nsign, integer, float, backend, runtime, prev)
    i = @index(Global)
    # (; terms) = model

    @inbounds begin
        # cell = cells[i]
        # @synchronize
        (; faces_range, volume) = cells[i]

        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            (; area, delta) = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]

            cIndex = nzval_index(colptr_array, rowval_array, i, i, ione)
            nIndex = nzval_index(colptr_array, rowval_array, nID, i, ione)

            # scheme code
            ap = term.sign*(-term.flux[fID] * area)/delta
            Atomix.@atomic nzval_array[cIndex] += ap
            Atomix.@atomic nzval_array[nIndex] += -ap
        end

    # scheme_scource loop
    # no scheme_source code for linear laplacian scheme

    end
end

@inline function schemes_and_sources!(
    term::Operator{F,P,I,Divergence{Linear}}, 
    nTerms, nSources, offset, fzero, ione, terms, rowval_array, colptr_array, nzval_array, cIndex, nIndex,  b, faces,
    cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
    backend, runtime, prev)  where {F,P,I}

    kernel! = schemes_divergence_linear!(backend)
    kernel!(term, nTerms, nSources, offset, fzero, ione, terms, rowval_array, colptr_array, nzval_array, cIndex, nIndex,  b, faces,
            cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
            backend, runtime, prev, ndrange = length(cells))
    KernelAbstractions.synchronize(backend)
end

@kernel function schemes_divergence_linear!(term, nTerms, nSources, offset, fzero, ione, terms,  rowval_array, colptr_array, nzval_array, cIndex, nIndex, b, faces, cells, cell_faces, cell_neighbours, cell_nsign, integer, float, backend, runtime, prev)
    i = @index(Global)
    # (; terms) = model

    @inbounds begin
        # cell = cells[i]
        # @synchronize
        (; faces_range, volume) = cells[i]

        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            (; centre) = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]

            # start = colptr_array[i]
            # # offset = findfirst(isequal(i),@view rowval_array[start:end]) - ione
            # offset = 0
            # for j in start:length(rowval_array)
            #     offset += 1
            #     if rowval_array[j] == i
            #         break
            #     end
            # end
            # cIndex = start + offset - ione
            cIndex = nzval_index(colptr_array, rowval_array, i, i, ione)

            # start = colptr_array[nID]
            # offset = findfirst(isequal(i),@view rowval_array[start:end]) - ione
            # offset = 0
            # for j in start:length(rowval_array)
            #     offset += 1
            #     if rowval_array[j] == i
            #         break
            #     end
            # end
            # nIndex = start + offset - ione
            nIndex = nzval_index(colptr_array, rowval_array, nID, i, ione)

            # scheme code
            xf = centre
            xC = cells[i].centre
            xN = cellN.centre
            weight = norm(xf - xC)/norm(xN - xC)
            one_minus_weight = one(eltype(weight)) - weight
            ap = term.sign*(term.flux[fID]*ns)
            Atomix.@atomic nzval_array[cIndex] += ap*one_minus_weight
            Atomix.@atomic nzval_array[nIndex] += ap*weight

        end

    # scheme_scource loop
    # no scheme_source code for divergence linear scheme

    end
end

@inline function schemes_and_sources!(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    nTerms, nSources, offset, fzero, ione, terms, rowval_array, colptr_array, nzval_array, cIndex, nIndex,  b, faces,
    cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
    backend, runtime, prev)  where {F,P,I}

    kernel! = schemes_divergence_upwind!(backend)
    kernel!(term, nTerms, nSources, offset, fzero, ione, terms, rowval_array, colptr_array, nzval_array, cIndex, nIndex,  b, faces,
            cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
            backend, runtime, prev, ndrange = length(cells))
    KernelAbstractions.synchronize(backend)
end

@kernel function schemes_divergence_upwind!(term, nTerms, nSources, offset, fzero, ione, terms,  rowval_array, colptr_array, nzval_array, cIndex, nIndex, b, faces, cells, cell_faces, cell_neighbours, cell_nsign, integer, float, backend, runtime, prev)
    i = @index(Global)
    # (; terms) = model

    @inbounds begin
        # cell = cells[i]
        # @synchronize
        (; faces_range, volume) = cells[i]

        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            # face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]

            # start = colptr_array[i]
            # # offset = findfirst(isequal(i),@view rowval_array[start:end]) - ione
            # offset = 0
            # for j in start:length(rowval_array)
            #     offset += 1
            #     if rowval_array[j] == i
            #         break
            #     end
            # end
            # cIndex = start + offset - ione
            cIndex = nzval_index(colptr_array, rowval_array, i, i, ione)

            # start = colptr_array[nID]
            # offset = findfirst(isequal(i),@view rowval_array[start:end]) - ione
            # offset = 0
            # for j in start:length(rowval_array)
            #     offset += 1
            #     if rowval_array[j] == i
            #         break
            #     end
            # end
            # nIndex = start + offset - ione
            nIndex = nzval_index(colptr_array, rowval_array, nID, i, ione)

            # scheme code
            ap = term.sign*(term.flux[fID]*ns)
            Atomix.@atomic nzval_array[cIndex] += max(ap, 0.0)
            Atomix.@atomic nzval_array[nIndex] += -max(-ap, 0.0)

        end

    # scheme_scource loop
    # no scheme_source code for divergence upwind scheme

    end
end

@inline function schemes_and_sources!(
    term::Operator{F,P,I,Si}, 
    nTerms, nSources, offset, fzero, ione, terms, rowval_array, colptr_array, nzval_array, cIndex, nIndex,  b, faces,
    cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
    backend, runtime, prev)  where {F,P,I}

    kernel! = schemes_si!(backend)
    kernel!(term, nTerms, nSources, offset, fzero, ione, terms, rowval_array, colptr_array, nzval_array, cIndex, nIndex,  b, faces,
            cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
            backend, runtime, prev, ndrange = length(cells))
    KernelAbstractions.synchronize(backend)            
end

@kernel function schemes_si!(term, nTerms, nSources, offset, fzero, ione, terms,  rowval_array, colptr_array, nzval_array, cIndex, nIndex, b, faces, cells, cell_faces, cell_neighbours, cell_nsign, integer, float, backend, runtime, prev)
    i = @index(Global)
    # (; terms) = model

    @inbounds begin
        # cell = cells[i]
        # @synchronize
        (; faces_range, volume) = cells[i]

        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            # face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]

            # start = colptr_array[i]
            # # offset = findfirst(isequal(i),@view rowval_array[start:end]) - ione
            # offset = 0
            # for j in start:length(rowval_array)
            #     offset += 1
            #     if rowval_array[j] == i
            #         break
            #     end
            # end
            # cIndex = start + offset - ione
            cIndex = nzval_index(colptr_array, rowval_array, i, i, ione)

            # start = colptr_array[nID]
            # offset = findfirst(isequal(i),@view rowval_array[start:end]) - ione
            # offset = 0
            # for j in start:length(rowval_array)
            #     offset += 1
            #     if rowval_array[j] == i
            #         break
            #     end
            # end
            # nIndex = start + offset - ione
            nIndex = nzval_index(colptr_array, rowval_array, nID, i, ione)

            # scheme code
            phi = term.phi
            # ap = max(flux, 0.0)
            # ab = min(flux, 0.0)*phi[cID]
            flux = term.sign*term.flux[i]*cell.volume # indexed with cID
            Atomix.@atomic nzval_array[cIndex] += flux # indexed with cIndex
            # flux = term.sign*term.flux[cID]*cell.volume*phi[cID]
            # b[cID] -= flux

        end

    # scheme_scource loop
    # no scheme_source code for si scheme

    end
end

@kernel function sources!(field, sign, cells, b)
    i = @index(Global)

    @inbounds begin
        # cell = cells[i]
        # @synchronize
        volume = cells[i].volume
        Atomix.@atomic b[i] += sign*field[i]*volume
    end
end

@kernel function set_b!(fzero, b)
    i = @index(Global)

    @inbounds begin
        # @synchronize
        b[i] = fzero
    end
end